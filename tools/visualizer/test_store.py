"""Independent checks of the tables through the visualizer's Python decoder.

This path shares no code with the Rust solver or the shakmaty-based verifier.
The numpy and zstandard packages read the .tbp/.tbz files. The python-chess package
judges legality and probes Syzygy.

Usage: python -I tools/visualizer/test_store.py TABLE_DIR SYZYGY_DIR [SAMPLES]
"""

from __future__ import annotations

import json
import random
import sys
import time
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import Store, chess_legal

TABLES = Path(sys.argv[1]) if len(sys.argv) > 1 else None
SYZYGY = Path(sys.argv[2]) if len(sys.argv) > 2 else None
SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
REPORT: dict = {}


def store() -> Store:
    if not hasattr(store, "cached"):
        store.cached = Store([TABLES], SYZYGY, None)
    return store.cached


class KnownPositions(unittest.TestCase):
    def probe(self, fen: str) -> dict:
        return store().probe(fen)

    def test_kqvk_mate_and_stalemate(self) -> None:
        mate = self.probe("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual((mate["s"], mate["w"], mate["r"], mate["n"]), ("ok", -2, 0, 0))
        stale = self.probe("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual((stale["w"], stale["n"]), (0, 0))

    def test_illegal_and_unsupported(self) -> None:
        self.assertEqual(self.probe("8/8/8/8/8/8/1kK5/Q7 w - - 0 1")["s"], "illegal")
        self.assertEqual(self.probe("8/8/8/8/8/8/k7/R6K w - - 0 1")["s"], "illegal")
        r = self.probe("8/8/8/8/8/8/kP6/7K w - - 0 1")
        self.assertEqual((r["s"], r["why"]), ("unsupported", "pawns"))
        self.assertEqual(self.probe("not a fen")["s"], "fen")

    def test_colour_swapped_material(self) -> None:
        # Black rook against bare king: KvKR, or KRvK with colours swapped.
        r = self.probe("8/8/3k4/8/8/8/8/r3K3 w - - 0 1")
        self.assertEqual((r["s"], r["w"]), ("ok", -2))
        self.assertIn(r["m"], ("KvKR", "KRvK"))
        if "KvKR" in store().tables and "KRvK" in store().tables:
            direct = store().tables["KvKR"].lookup(False, [chess.E1, chess.A1], chess.D6)
            swapped = store().tables["KRvK"].lookup(True, [chess.D6, chess.A1], chess.E1)
            self.assertEqual(direct, swapped)

    def test_move_values_match_child_probes(self) -> None:
        fen = "8/8/8/8/8/2k5/1r6/KQ6 w - - 0 1"
        r = self.probe(fen)
        self.assertEqual(r["s"], "ok")
        board = chess.Board(fen)
        moves = sorted(board.legal_moves, key=chess.Move.uci)
        self.assertEqual(r["n"], len(moves))
        self.assertEqual(max(v[0] for v in r["v"]), r["w"])
        for move, value in zip(moves, r["v"], strict=True):
            board.push(move)
            child = store().probe(board.fen())
            board.pop()
            self.assertEqual(child["s"], "ok")
            self.assertEqual(-child["w"], value[0], move.uci())

    def test_best_line_is_consistent(self) -> None:
        fen = "8/8/8/8/4k3/8/1Q6/K7 b - - 0 1"
        line = store().best_line(fen)["u"]
        board = chess.Board(fen)
        for uci in line:
            board.push_uci(uci)
        self.assertTrue(board.is_checkmate())
        self.assertEqual(len(line), store().probe(fen)["r"])


class SampledAgreement(unittest.TestCase):
    """Random positions from every table: legality against python-chess, WDL against Syzygy."""

    def test_every_table(self) -> None:
        rng = random.Random(20261014)  # noqa: S311 - reproducible samples
        s = store()
        for name, table in sorted(s.tables.items()):
            start = time.perf_counter()
            checked = legal = syzygy = 0
            while checked < SAMPLES:
                word = rng.randrange(table.words)
                bk = rng.randrange(64)
                black_to_move, squares = table.decode(word)
                if len(set(squares)) != len(squares) or bk in squares:
                    continue
                board = chess.Board(None)
                for (white, role), square in zip(table.pieces, squares, strict=True):
                    board.set_piece_at(square, chess.Piece(role, white))
                board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
                board.turn = not black_to_move
                checked += 1
                if table.knows_legality:
                    wdl, _ = table.value(word, bk)
                    with self.subTest(material=name, fen=board.fen()):
                        self.assertEqual(wdl is not None, chess_legal(board))
                elif chess_legal(board):
                    wdl, _ = table.value(word, bk)
                else:
                    wdl = None
                if wdl is None:
                    continue
                legal += 1
                reference = s.syzygy_wdl(board)
                if reference is None:
                    continue
                syzygy += 1
                with self.subTest(material=name, fen=board.fen()):
                    self.assertEqual(wdl // 2, (reference > 0) - (reference < 0))
            REPORT[name] = {
                "sampled": checked,
                "legal": legal,
                "syzygy_compared": syzygy,
                "seconds": round(time.perf_counter() - start, 2),
            }


if __name__ == "__main__":
    if TABLES is None or SYZYGY is None:
        raise SystemExit(__doc__)
    sys.argv = sys.argv[:1]
    result = unittest.main(exit=False, verbosity=1).result
    print(json.dumps({"samples_per_table": SAMPLES, "tables": REPORT}, indent=1))
    sys.exit(0 if result.wasSuccessful() else 1)
