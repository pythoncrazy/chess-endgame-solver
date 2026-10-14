"""Local visualizer for pawnless tablebases (*.tbz format v3/v4, *.tbp v1/v2).

Usage:
  python -I tools/visualizer/server.py --tables DIR [--tables DIR ...]
      [--syzygy DIR ...] [--logs DIR] [--host 127.0.0.1] [--port 8642]

Network budget: the page is one self-contained document (CSS, JS and the piece
sprite inline) revalidated by ETag; fonts are content-addressed and immutable.
API replies are compact JSON, gzip-compressed, over keep-alive connections. The
client generates pawnless moves itself, so a probe returns only the value of
each move, in UCI order. Tables are memory-mapped. A packed position costs one
cached block per section (WDL, rank); a v4 black-to-move rank is derived from
its white-to-move children; a v2 position costs one record. Values are never
guessed.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import http.server
import json
import lzma
import mmap
import random
import re
import socket
import subprocess
import threading
import time
import urllib.parse
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.svg
import chess.syzygy
import numpy as np
import zstandard

STATIC = Path(__file__).resolve().parent / "static"
UNRESOLVED = 0xFFFF
ROLE_ORDER = "QRBN"
ROLES = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}
GZIP_MIN = 200


class TableFormatError(ValueError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"{path}: not a pawnless table file")


def skeleton(name: str) -> list[tuple[bool, int]]:
    """Skeleton order of a material name: white king, white pieces, black pieces."""
    white, black = name.split("v")
    pieces = [(True, chess.KING)]
    pieces += [(True, ROLES[c]) for c in white[1:]]
    pieces += [(False, ROLES[c]) for c in black[1:]]
    return pieces


class Layout:
    """Full working layout: word = turn * 64^m + skeleton squares (white king first)."""

    header: dict
    pieces: list[tuple[bool, int]]
    knows_legality = True

    @property
    def name(self) -> str:
        return self.header["material"]

    def encode(self, black_to_move: bool, squares: list[int]) -> int:
        word = int(black_to_move)
        for square in squares:
            word = word * 64 + square
        return word

    def decode(self, word: int) -> tuple[bool, list[int]]:
        squares = []
        for _ in self.pieces:
            squares.append(word % 64)
            word //= 64
        return word == 1, squares[::-1]

    def value(self, word: int, bk: int) -> tuple[int | None, int | None]:
        return self.lookup(*self.decode(word), bk)

    def lookup(
        self, black_to_move: bool, squares: list[int], bk: int
    ) -> tuple[int | None, int | None]:
        raise NotImplementedError


@dataclass
class Table(Layout):
    """A memory-mapped table. Version 2 keeps one record per word, so a probe
    is one contiguous read; version 1 (three planes) is still accepted."""

    path: Path
    header: dict
    words: int
    version: int
    data: np.ndarray
    ranked: bool
    rank_width: int
    pieces: list[tuple[bool, int]] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path) -> Table:
        with path.open("rb") as f:
            magic = f.read(8)
            if magic not in (b"TBPAWNL1", b"TBPAWNL2"):
                raise TableFormatError(path)
            length = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(length))
        words, base = header["words"], 16 + length
        version = 2 if magic == b"TBPAWNL2" else 1
        ranked = len(header["planes"]) == (3 if version == 2 else 4)
        rank_width = 0
        if ranked:
            rank_width = 1 if version == 2 and header["planes"][2] == "rank:u8x64" else 2
        if version == 2:
            record = 16 + 64 * rank_width
            data = np.memmap(path, dtype=np.uint8, mode="r", offset=base, shape=(words, record))
        else:
            data = np.memmap(path, dtype=np.uint8, mode="r", offset=base)
        return cls(
            path, header, words, version, data, ranked, rank_width, skeleton(header["material"])
        )

    def lookup(
        self, black_to_move: bool, squares: list[int], bk: int
    ) -> tuple[int | None, int | None]:
        """(WDL for the side to move, proof rank) or (None, None) if illegal."""
        word = self.encode(black_to_move, squares)
        bit = 1 << bk
        if self.version == 2:
            rec = bytes(self.data[word])  # the single read
            win = int.from_bytes(rec[0:8], "little")
            loss = int.from_bytes(rec[8:16], "little")
            if win & loss & bit:
                return None, None
            rank = None
            if self.rank_width == 1:
                rank = UNRESOLVED if rec[16 + bk] == 255 else rec[16 + bk]
            elif self.rank_width == 2:
                rank = int.from_bytes(rec[16 + 2 * bk : 18 + 2 * bk], "little")
        else:
            d, n = self.data, self.words

            def u64(i: int) -> int:
                return int.from_bytes(bytes(d[8 * i : 8 * i + 8]), "little")

            if not u64(word) & bit:
                return None, None
            win, loss = u64(n + word), u64(2 * n + word)
            rank = None
            if self.ranked:
                at = 24 * n + 2 * (word * 64 + bk)
                rank = int.from_bytes(bytes(d[at : at + 2]), "little")
        wdl = 2 if win & bit else -2 if loss & bit else 0
        return wdl, None if rank in (None, UNRESOLVED) else rank


# Formats v3/v4 (rust/src/packed.rs, rust/TBZ_FORMAT.md): one position per symmetry class.
TRIANGLE = (0, 1, 2, 3, 9, 10, 11, 18, 19, 27)
TRIANGLE_INDEX = {s: i for i, s in enumerate(TRIANGLE)}
LOWER_RIGHT = 0x80C0E0F0F8FCFEFF


def _symmetry(g: int, s: int) -> int:
    """Bit 0 transpose (a1-h8), then bit 1 file mirror, then bit 2 rank mirror."""
    if g & 1:
        s = ((s & 7) << 3) | (s >> 3)
    if g & 2:
        s ^= 7
    if g & 4:
        s ^= 56
    return s


SYMMETRY = [[_symmetry(g, s) for s in range(64)] for g in range(8)]
TRANSPOSE = SYMMETRY[1]
TO_TRIANGLE = [next(g for g in range(8) if SYMMETRY[g][s] in TRIANGLE_INDEX) for s in range(64)]
WDL_CODE = (0, 2, -2, 2)


class BlockCache:
    """Decompressed blocks shared by every packed table, least recently used out."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.blocks: OrderedDict[tuple, bytes] = OrderedDict()
        self.lock = threading.Lock()
        self.local = threading.local()
        self.hits = self.misses = 0

    def get(self, key: tuple, frame: memoryview, size: int, codec: str = "zstd") -> bytes:
        with self.lock:
            block = self.blocks.get(key)
            if block is not None:
                self.blocks.move_to_end(key)
                self.hits += 1
                return block
            self.misses += 1
        if codec == "lzma":
            block = lzma.LZMADecompressor(lzma.FORMAT_ALONE).decompress(frame, max_length=size)
        else:
            if not hasattr(self.local, "zstd"):
                self.local.zstd = zstandard.ZstdDecompressor()
            block = self.local.zstd.decompress(frame, max_output_size=size)
        if len(block) != size:
            raise TableFormatError(Path(str(key[0])))
        with self.lock:
            self.blocks[key] = block
            if len(self.blocks) > self.capacity:
                self.blocks.popitem(last=False)
        return block


BLOCKS = BlockCache(2048)


class PackedTable(Layout):
    """A v3 or v4 table. Legality is not stored; callers judge it with python-chess."""

    knows_legality = False

    def __init__(self, path: Path) -> None:
        with path.open("rb") as f:
            self.map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        if self.map[:8] not in (b"TBPAWNZ1", b"TBPAWNZ2"):
            raise TableFormatError(path)
        length = int.from_bytes(self.map[8:16], "little")
        self.path = path
        self.header = json.loads(self.map[16 : 16 + length])
        self.pieces = skeleton(self.header["material"])
        self.m = len(self.pieces)
        self.per = 64 ** (self.m - 1)
        self.words = 2 * 64**self.m
        self.rank_width = self.header["rank_width"]
        self.ranked = self.rank_width > 0
        self.derived = self.header.get("rank_sides", "both") == "white"
        self.block = self.header["block_bytes"]
        self.groups = []
        start = 1
        for i in range(2, self.m + 1):
            if i == self.m or self.pieces[i] != self.pieces[start]:
                if i - start >= 2:
                    self.groups.append((start, i))
                start = i
        view = memoryview(self.map)
        self.sections = {}
        for s in self.header["sections"]:
            at = 16 + length + s["offset"]
            index = np.frombuffer(view[at : at + 8 * (s["blocks"] + 1)], dtype="<u8")
            frames = at + 8 * (s["blocks"] + 1)
            self.sections[s["name"]] = (frames, index, s["raw_bytes"], s.get("codec", "zstd"))
        self.view = view

    def _mapped(self, g: int, squares: list[int]) -> list[int]:
        table = SYMMETRY[g]
        out = [table[s] for s in squares]
        for a, b in self.groups:
            out[a:b] = sorted(out[a:b])
        return out

    def position(self, black_to_move: bool, squares: list[int], bk: int) -> int:
        """Stored index: ((turn*10 + triangle(wk)) * 64^(m-1) + rest) * 64 + bk."""
        g = TO_TRIANGLE[squares[0]]
        stored = self._mapped(g, squares)
        t = SYMMETRY[g][bk]
        if stored[0] % 9 == 0:  # white king on the long diagonal
            transposed = self._mapped(1, stored)
            if transposed < stored:
                stored, t = transposed, TRANSPOSE[t]
            elif transposed == stored and not LOWER_RIGHT >> t & 1:
                t = TRANSPOSE[t]
        rest = 0
        for s in stored[1:]:
            rest = rest * 64 + s
        word = (int(black_to_move) * 10 + TRIANGLE_INDEX[stored[0]]) * self.per + rest
        return word * 64 + t

    def _byte(self, section: str, at: int) -> bytes:
        frames, index, raw, codec = self.sections[section]
        b = at // self.block
        frame = self.view[frames + int(index[b]) : frames + int(index[b + 1])]
        size = min(self.block, raw - b * self.block)
        block = BLOCKS.get((self.path, section, b), frame, size, codec)
        return block[at % self.block : at % self.block + 2]

    def _code(self, p: int) -> int:
        return self._byte("wdl", p // 4)[0] >> (2 * (p % 4)) & 3

    def _rank(self, p: int) -> int:
        data = self._byte("rank", p * self.rank_width)
        return data[0] if self.rank_width == 1 else int.from_bytes(data[:2], "little")

    def _derive_black(self, squares: list[int], bk: int, win: bool) -> int:
        """Rank of a decisive black-to-move position from its white-to-move
        children: a win is one more than its fastest losing child, a loss one
        more than its slowest child; captures (lower material) count as 0."""
        board = chess.Board(None)
        board.turn = chess.BLACK
        for (white, role), square in zip(self.pieces, squares, strict=True):
            board.set_piece_at(square, chess.Piece(role, white))
        board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
        best, moved = None, False
        for move in board.legal_moves:
            moved = True
            if board.piece_at(move.to_square) is not None:
                continue
            child, child_bk = list(squares), bk
            if move.from_square == bk:
                child_bk = move.to_square
            else:
                child[child.index(move.from_square)] = move.to_square
            p = self.position(False, child, child_bk)
            if win:
                if self._code(p) == 2:
                    r = self._rank(p)
                    best = r if best is None else min(best, r)
            else:
                best = max(best or 0, self._rank(p))
        if win:
            if best is None:
                raise TableFormatError(self.path)
            return best + 1
        return (best or 0) + 1 if moved else 0

    def lookup(
        self, black_to_move: bool, squares: list[int], bk: int
    ) -> tuple[int | None, int | None]:
        """(WDL for the side to move, proof rank) of a legal position."""
        p = self.position(black_to_move, squares, bk)
        code = self._code(p)
        derived = black_to_move and self.derived
        if code == 3 and not derived:
            raise TableFormatError(self.path)
        wdl = WDL_CODE[code]
        rank = None
        if wdl and self.ranked:
            if not derived:
                rank = self._rank(p)
            elif code == 3:
                rank = 1
            else:
                rank = self._derive_black(squares, bk, wdl == 2)
        return wdl, rank


def open_table(path: Path) -> Layout:
    return PackedTable(path) if path.suffix == ".tbz" else Table.open(path)


def material_name(board: chess.Board, white: bool) -> str:
    def side(color: chess.Color) -> str:
        return "K" + "".join(c * len(board.pieces(ROLES[c], color)) for c in ROLE_ORDER)

    us, them = (chess.WHITE, chess.BLACK) if white else (chess.BLACK, chess.WHITE)
    return f"{side(us)}v{side(them)}"


def chess_legal(board: chess.Board) -> bool:
    """python-chess legality, accepting check patterns that cannot arise in play
    (the tables and the shakmaty verifier accept them too)."""
    ignored = chess.STATUS_IMPOSSIBLE_CHECK | chess.STATUS_TOO_MANY_CHECKERS
    return board.status() & ~ignored == chess.STATUS_VALID


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


class Store:
    def __init__(
        self, table_dirs: list[Path], syzygy: Path | list[Path] | None, logs: Path | None
    ) -> None:
        self.tables: dict[str, Layout] = {}
        for directory in table_dirs:
            paths = sorted(directory.glob("*.tbz")) + sorted(
                directory.glob("*.tbp"), key=lambda p: ".gpu." not in p.name
            )
            for path in paths:
                table = open_table(path)
                self.tables.setdefault(table.name, table)
        dirs = [syzygy] if isinstance(syzygy, Path) else list(syzygy or [])
        self.syzygy = chess.syzygy.open_tablebase(str(dirs[0])) if dirs else None
        for directory in dirs[1:]:
            self.syzygy.add_directory(str(directory))
        self.syzygy_lock = threading.Lock()
        self.logs = logs
        self._load_evidence()

    # Evidence ---------------------------------------------------------------------
    def _load_evidence(self) -> None:
        self.solves: dict[str, dict] = {}
        self.verifications: dict[str, list[dict]] = {}
        self.reference_checks: dict[str, dict] = {}
        self.dtm_checks: dict[str, dict] = {}
        if not self.logs:
            return
        for path in sorted(self.logs.glob("gpu-solve-*.stdout")):
            for t in _json(path).get("tables", []):
                timing = t.get("timing", {})
                rounds = timing.get("rounds", [])
                served = self.tables.get(t["material"])
                if t["material"] in self.solves and (
                    served is None or t["payload_sha256"] != served.header["payload_sha256"]
                ):
                    continue
                self.solves.setdefault(t["material"], {}).update(
                    {
                        "log": path.name,
                        "digest": t["payload_sha256"],
                        "mode": timing.get("mode"),
                        "counts": [t["counts_white_to_move"], t["counts_black_to_move"]],
                        "changed": [r["changed_words"] for r in rounds],
                        "sweep_ms": [
                            round(max(r.get("sweep_event_ms") or 0, 0), 3) for r in rounds
                        ],
                        "seconds": timing.get("rounds_seconds_total"),
                        "bytes": timing.get("device_bytes_allocated"),
                    }
                )
        for path in sorted(self.logs.glob("verify-*.stdout")):
            for t in _json(path).get("tables", []):
                self.verifications.setdefault(t["payload_sha256"], []).append(
                    {**t, "log": path.name}
                )
        online = self.logs.parent / "results" / "online"
        names = {"agree": "a", "DIFFER": "d", "recorded-separately": "s"}
        for row in _json(online / "online_comparison.json").get("rows", []):
            entry = self.reference_checks.setdefault(
                row["material"], {"a": 0, "d": 0, "u": 0, "s": 0}
            )
            entry[names.get(row["result"], "u")] += 1
        dtm = _json(online / "online_dtm_comparison.json")
        if dtm.get("summary"):
            rows = dtm.get("rows", [])
            self.dtm_checks[dtm["summary"]["payload_sha256"]] = {
                "a": sum(r["result"] == "agree" for r in rows),
                "n": len(rows),
            }

    def verification_state(self, table: Table) -> tuple[str, int]:
        runs = self.verifications.get(table.header["payload_sha256"], [])
        if not runs:
            return "pending", 0
        return ("pass" if all(r["status"] == "pass" for r in runs) else "fail"), len(runs)

    def exact_dtm(self, name: str) -> bool:
        """Proof ranks equal distance to mate when every capture leads to a material
        without decisive positions (those children then correctly count as rank 0)."""
        pieces = skeleton(name)
        for k in range(1, len(pieces)):
            rest = pieces[:k] + pieces[k + 1 :]
            white = "".join(chess.piece_symbol(r).upper() for w, r in rest[1:] if w)
            black = "".join(chess.piece_symbol(r).upper() for w, r in rest[1:] if not w)
            solve = self.solves.get(f"K{white}vK{black}")
            if solve is None or any(c["win"] or c["loss"] for c in solve["counts"]):
                return False
        return self.tables[name].ranked

    def materials(self) -> dict:
        """Compact table list: one row per table, fixed column order."""
        rows = []
        for name, table in sorted(self.tables.items(), key=lambda kv: (len(kv[0]), kv[0])):
            solve = self.solves.get(name, {})
            counts = solve.get("counts")
            state, runs = self.verification_state(table)
            gen = re.sub(
                r"git:([0-9a-f]{8})[0-9a-f]+", r"git:\1", table.header.get("generator", "")
            )
            rows.append(
                [
                    name,
                    int(table.ranked),
                    int(self.exact_dtm(name)),
                    table.header["rounds"],
                    [[c["legal"], c["win"], c["draw"], c["loss"]] for c in counts]
                    if counts
                    else None,
                    state,
                    runs,
                    gen,
                ]
            )
        return {
            "cols": ["name", "ranked", "exactDtm", "rounds", "counts", "verified", "runs", "gen"],
            "rows": rows,
        }

    def evidence(self, name: str) -> dict:
        table = self.tables.get(name)
        if table is None:
            return {}
        digest = table.header["payload_sha256"]
        keep = (
            "status",
            "scope",
            "legal_positions",
            "syzygy_wdl_compared",
            "proof_obligations_checked",
            "symmetry_violating_words",
            "proof_ranks_present",
            "first_failure",
            "seconds",
            "log",
        )
        return {
            "digest": digest,
            "generator": table.header.get("generator", ""),
            "solve": self.solves.get(name),
            "verify": [{k: v.get(k) for k in keep} for v in self.verifications.get(digest, [])],
            "ref": self.reference_checks.get(name),
            "dtm": self.dtm_checks.get(digest),
        }

    # Probing ----------------------------------------------------------------------
    def locate(self, board: chess.Board) -> tuple[Layout, bool, list[int], int] | None:
        """Table, turn, skeleton squares and black king of a pawnless board (colours
        swapped if needed)."""
        for flipped in (False, True):
            table = self.tables.get(material_name(board, white=not flipped))
            if table is None:
                continue
            squares, used = [], 0
            for white, role in table.pieces:
                free = int(board.pieces_mask(role, white != flipped)) & ~used
                square = (free & -free).bit_length() - 1
                used |= 1 << square
                squares.append(square)
            bk = board.king(chess.WHITE if flipped else chess.BLACK)
            black_to_move = (board.turn == chess.BLACK) != flipped
            return table, black_to_move, squares, bk
        return None

    def unsupported(self, board: chess.Board) -> str | None:
        if board.pawns:
            return "pawns"
        if board.castling_rights:
            return "castling"
        if any(len(board.pieces(chess.KING, c)) != 1 for c in chess.COLORS):
            return "kings"
        if chess.popcount(board.occupied) > 6:
            return "pieces"
        return None

    def value(self, board: chess.Board) -> tuple[int | None, int | None]:
        found = self.locate(board)
        if found is None:
            return None, None
        table, *position = found
        return table.lookup(*position)

    def syzygy_wdl(self, board: chess.Board) -> int | None:
        if self.syzygy is None:
            return None
        zero = board.copy(stack=False)
        zero.halfmove_clock = 0
        try:
            with self.syzygy_lock:
                return self.syzygy.probe_wdl(zero)
        except (KeyError, chess.syzygy.MissingTableError):
            return None

    def probe(self, fen: str) -> dict:
        """Compact probe. Keys: s status, m material, w WDL, r rank, z Syzygy WDL,
        v [[wdl, rank], ...] per legal move in sorted UCI order, n move count,
        d indices whose child disagrees with Syzygy, x legality disagreement."""
        start = time.perf_counter()
        try:
            board = chess.Board(fen)
        except ValueError:
            return {"s": "fen"}
        reason = self.unsupported(board)
        if reason:
            return {"s": "unsupported", "why": reason}
        found = self.locate(board)
        if found is None:
            return {"s": "missing", "m": material_name(board, white=True)}
        table = found[0]
        legal = chess_legal(board)
        wdl, rank = table.lookup(*found[1:]) if legal or table.knows_legality else (None, None)
        out: dict = {"m": table.name}
        if wdl is None or not legal:
            out["s"] = "illegal"
            if table.knows_legality and (wdl is None) != (not legal):
                out["x"] = 1
            return out
        out |= {"s": "ok", "w": wdl}
        if rank is not None:
            out["r"] = rank
        syzygy = self.syzygy_wdl(board)
        if syzygy is not None:
            out["z"] = syzygy
        values, disagree = [], []
        for i, move in enumerate(sorted(board.legal_moves, key=chess.Move.uci)):
            board.push(move)
            child_wdl, child_rank = self.value(board)
            if child_wdl is not None:
                child_syzygy = self.syzygy_wdl(board)
                if child_syzygy is not None and sign(child_syzygy) != child_wdl // 2:
                    disagree.append(i)
            board.pop()
            values.append(
                [None if child_wdl is None else -child_wdl, child_rank]
                if child_rank is not None
                else [None if child_wdl is None else -child_wdl]
            )
        out |= {"v": values, "n": len(values)}
        if disagree:
            out["d"] = disagree
        out["ms"] = round((time.perf_counter() - start) * 1000, 2)
        return out

    def best_line(self, fen: str, limit: int = 120) -> dict:
        """Best play from a decided position: fastest win, longest defence."""
        try:
            board = chess.Board(fen)
        except ValueError:
            return {"s": "fen"}
        line = []
        for _ in range(limit):
            if board.is_game_over() or self.unsupported(board):
                break
            wdl, _ = self.value(board)
            if wdl in (None, 0):
                break
            best, best_key = None, None
            for move in board.legal_moves:
                board.push(move)
                child, rank = self.value(board)
                mate = board.is_checkmate()
                board.pop()
                if child is None:
                    continue
                r = rank if rank is not None else 1 << 20
                key = (0 if mate else 1, r) if -child == 2 else (2, -r)
                if -child == wdl and (best_key is None or key < best_key):
                    best, best_key = move, key
            if best is None:
                break
            line.append(best.uci())
            board.push(best)
        return {"s": "ok", "u": line}

    def random_position(self, material: str, turn: str, want: str) -> dict:
        table = self.tables.get(material)
        if table is None:
            return {"s": "missing", "m": material}
        rng = random.SystemRandom()
        goal = {"win": 2, "draw": 0, "loss": -2}.get(want)
        for _ in range(200_000):
            word = rng.randrange(table.words // 2)
            if turn == "b" or (turn == "any" and rng.random() < 0.5):
                word += table.words // 2
            bk = rng.randrange(64)
            black_to_move, squares = table.decode(word)
            if len({*squares, bk}) != len(squares) + 1:
                continue
            board = chess.Board(None)
            for (white, role), square in zip(table.pieces, squares, strict=True):
                board.set_piece_at(square, chess.Piece(role, white))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.turn = not black_to_move
            if not chess_legal(board):
                continue
            wdl, _ = table.lookup(black_to_move, squares, bk)
            if wdl is None or (goal is not None and wdl != goal):
                continue
            return {"s": "ok", "f": board.fen()}
        return {"s": "none"}


# Assets -------------------------------------------------------------------------------
def piece_sprite() -> str:
    """cburnett-style pieces from python-chess as one inline <symbol> sprite."""
    symbols = []
    for color in "wb":
        for role in "KQRBN":
            symbol = role if color == "w" else role.lower()
            svg = chess.svg.piece(chess.Piece.from_symbol(symbol))
            inner = re.sub(r"^<svg[^>]*>|</svg>$", "", svg.strip())
            symbols.append(f'<symbol id="{color}{role}" viewBox="0 0 45 45">{inner}</symbol>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" hidden>{"".join(symbols)}</svg>'


ESBUILD = ["npx", "--no-install", "esbuild@0.25.10", "--minify"]


def minify(source: str, loader: str) -> tuple[str, bool]:
    """Minify with esbuild when it is installed locally; otherwise ship the source."""
    try:
        done = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            [*ESBUILD, f"--loader={loader}"],
            input=source,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return source, False
    return done.stdout, True


def build_assets() -> tuple[bytes, dict[str, bytes]]:
    """The single page (CSS, JS, sprite inline) and content-addressed font files."""
    fonts = {}
    css = (STATIC / "app.css").read_text()
    for path in sorted((STATIC / "fonts").glob("*.woff2")):
        data = path.read_bytes()
        name = f"{path.stem}.{hashlib.sha256(data).hexdigest()[:12]}.woff2"
        fonts[name] = data
        css = css.replace(f"/fonts/{path.name}", f"/fonts/{name}")
    css, css_min = minify(css, "css")
    js, js_min = minify((STATIC / "app.js").read_text(), "js")
    state = {True: "minified", False: "source"}
    print(f"assets: css {state[css_min]}, js {state[js_min]}")
    page = (STATIC / "index.html").read_text()
    page = page.replace("<!--CSS-->", f"<style>{css}</style>")
    page = page.replace("<!--SPRITE-->", piece_sprite())
    page = page.replace("<!--JS-->", f'<script type="module">{js}</script>')
    return page.encode(), fonts


class V6Server(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: no reconnect per request
    store: Store
    page: bytes
    page_gz: bytes
    page_etag: str
    fonts: dict[str, bytes]

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def reply(
        self,
        body: bytes,
        kind: str,
        cache: str,
        etag: str | None = None,
        compressed: bytes | None = None,
    ) -> None:
        if etag and self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        gz = "gzip" in self.headers.get("Accept-Encoding", "")
        if gz and compressed is None and len(body) >= GZIP_MIN and not kind.startswith("font/"):
            compressed = gzip.compress(body, compresslevel=6, mtime=0)
        payload = compressed if gz and compressed is not None else body
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", cache)
        self.send_header("Vary", "Accept-Encoding")
        if payload is compressed:
            self.send_header("Content-Encoding", "gzip")
        if etag:
            self.send_header("ETag", etag)
        self.end_headers()
        self.wfile.write(payload)

    def api(self, data: dict) -> None:
        body = json.dumps(data, separators=(",", ":")).encode()
        self.reply(body, "application/json", "no-store")

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        q = dict(urllib.parse.parse_qsl(url.query))
        s = self.store
        if url.path == "/":
            self.reply(
                self.page, "text/html; charset=utf-8", "no-cache", self.page_etag, self.page_gz
            )
        elif url.path == "/api/probe":
            self.api(s.probe(q.get("fen", "")))
        elif url.path == "/api/line":
            self.api(s.best_line(q.get("fen", "")))
        elif url.path == "/api/random":
            self.api(
                s.random_position(q.get("m", ""), q.get("turn", "any"), q.get("result", "any"))
            )
        elif url.path == "/api/materials":
            body = json.dumps(s.materials(), separators=(",", ":")).encode()
            etag = f'"{hashlib.sha256(body).hexdigest()[:16]}"'
            self.reply(body, "application/json", "no-cache", etag)
        elif url.path == "/api/evidence":
            self.api(s.evidence(q.get("m", "")))
        elif url.path.startswith("/fonts/") and url.path[7:] in self.fonts:
            self.reply(
                self.fonts[url.path[7:]], "font/woff2", "public, max-age=31536000, immutable"
            )
        else:
            self.reply(b"not found", "text/plain", "no-store")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, action="append", required=True)
    parser.add_argument("--syzygy", type=Path, action="append")
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument(
        "--host",
        action="append",
        help="bind address; repeat for several (IPv4 and IPv6); default loopback",
    )
    args = parser.parse_args()
    Handler.store = Store(args.tables, args.syzygy, args.logs)
    Handler.page, Handler.fonts = build_assets()
    Handler.page_gz = gzip.compress(Handler.page, compresslevel=9, mtime=0)
    Handler.page_etag = f'"{hashlib.sha256(Handler.page).hexdigest()[:16]}"'
    servers = []
    for host in args.host or ["127.0.0.1"]:
        server_class = V6Server if ":" in host else http.server.ThreadingHTTPServer
        servers.append(server_class((host, args.port), Handler))
        shown = f"[{host}]" if ":" in host else host
        print(f"{len(Handler.store.tables)} tables; http://{shown}:{args.port}/", flush=True)
    for server in servers[1:]:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
