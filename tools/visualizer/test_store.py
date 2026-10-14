"""Independent checks of the tables through the visualizer's Python decoder.

This path shares no code with the Rust solver or the shakmaty-based verifier:
numpy reads the .tbp planes, python-chess judges legality and probes Syzygy.

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
        self.assertEqual((mate["status"], mate["wdl"], mate["rank"]), ("ok", -2, 0))
        self.assertTrue(mate["checkmate"])
        stale = self.probe("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual((stale["wdl"], stale["stalemate"]), (0, True))

    def test_illegal_and_unsupported(self) -> None:
        self.assertEqual(self.probe("8/8/8/8/8/8/1kK5/Q7 w - - 0 1")["status"], "illegal")
        self.assertEqual(self.probe("8/8/8/8/8/8/k7/R6K w - - 0 1")["status"], "illegal")
        self.assertEqual(self.probe("8/8/8/8/8/8/kP6/7K w - - 0 1")["status"], "unsupported")
        self.assertEqual(self.probe("not a fen")["status"], "invalid-fen")

    def test_colour_swapped_material(self) -> None:
        # White rook vs bare king is stored as KvKR with colours swapped.
        r = self.probe("8/8/3k4/8/8/8/8/R3K3 b - - 0 1")
        self.assertEqual((r["status"], r["material"], r["wdl"]), ("ok", "KvKR", -2))

    def test_move_list_children_are_consistent(self) -> None:
        r = self.probe("8/8/8/8/8/2k5/1r6/KQ6 w - - 0 1")
        self.assertEqual(r["status"], "ok")
        best = max(m["wdl"] for m in r["moves"])
        self.assertEqual(best, r["wdl"])
        for m in r["moves"]:
            child = store().probe(m["fen"])
            self.assertEqual(child["status"], "ok")
            self.assertEqual(-child["wdl"], m["wdl"], m["san"])


class SampledAgreement(unittest.TestCase):
    """Random positions from every table: legality vs python-chess, WDL vs Syzygy."""

    def test_every_table(self) -> None:
        rng = random.Random(20261014)  # noqa: S311 - reproducible sampling
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
                wdl, _ = table.value(word, bk)
                checked += 1
                with self.subTest(material=name, fen=board.fen()):
                    self.assertEqual(wdl is not None, chess_legal(board))
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
