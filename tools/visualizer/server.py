"""Local visualizer for pawnless working-layout tablebases (*.tbp).

Usage:
  python -I tools/visualizer/server.py --tables DIR [--tables DIR ...]
      [--syzygy DIR] [--logs DIR] [--host 127.0.0.1] [--port 8642]

Tables are memory-mapped, so a probe reads a few words even from a 51.5 GB file.
Values are never guessed: positions outside the supported domain report why.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import random
import threading
import time
import typing
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.svg
import chess.syzygy
import numpy as np

STATIC = Path(__file__).resolve().parent / "static"
MAGIC = b"TBPAWNL1"
UNRESOLVED = 0xFFFF
ROLE_ORDER = "QRBN"
ROLES = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT}


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


@dataclass
class Table:
    path: Path
    header: dict
    words: int
    planes: np.ndarray  # (3, words): valid, win, loss
    rank: np.ndarray | None
    pieces: list[tuple[bool, int]] = field(default_factory=list)

    @classmethod
    def open(cls, path: Path) -> Table:
        with path.open("rb") as f:
            if f.read(8) != MAGIC:
                raise TableFormatError(path)
            length = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(length))
        words = header["words"]
        base = 16 + length
        planes = np.memmap(path, dtype="<u8", mode="r", offset=base, shape=(3, words))
        rank = None
        if len(header["planes"]) == 4:
            offset = base + 24 * words
            rank = np.memmap(path, dtype="<u2", mode="r", offset=offset, shape=(words * 64,))
        return cls(path, header, words, planes, rank, skeleton(header["material"]))

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
        bit = 1 << bk
        if not int(self.planes[0, word]) & bit:
            return None, None
        wdl = 2 if int(self.planes[1, word]) & bit else -2 if int(self.planes[2, word]) & bit else 0
        rank = None
        if self.rank is not None:
            r = int(self.rank[word * 64 + bk])
            rank = None if r == UNRESOLVED else r
        return wdl, rank


def material_name(board: chess.Board, white: bool) -> str:
    def side(color: chess.Color) -> str:
        return "K" + "".join(c * len(board.pieces(ROLES[c], color)) for c in ROLE_ORDER)

    us, them = (chess.WHITE, chess.BLACK) if white else (chess.BLACK, chess.WHITE)
    return f"{side(us)}v{side(them)}"


class Store:
    def __init__(self, table_dirs: list[Path], syzygy: Path | None, logs: Path | None) -> None:
        self.tables: dict[str, Table] = {}
        for directory in table_dirs:
            for path in sorted(directory.glob("*.tbp"), key=lambda p: ".gpu." not in p.name):
                table = Table.open(path)
                self.tables.setdefault(table.header["material"], table)
        self.syzygy = chess.syzygy.open_tablebase(str(syzygy)) if syzygy else None
        self.syzygy_lock = threading.Lock()
        self.logs = logs
        self.evidence = self._evidence()

    # Evidence from solver and verifier logs ------------------------------------------
    def _evidence(self) -> dict:
        solves: dict[str, dict] = {}
        verifications: dict[str, dict] = {}
        reference_checks: dict[str, dict] = {}
        dtm_checks: dict[str, dict] = {}
        if self.logs:
            for path in sorted(self.logs.glob("gpu-solve-*.stdout")):
                for t in _json(path).get("tables", []):
                    rounds = t.get("timing", {}).get("rounds", [])
                    solves.setdefault(t["material"], {}).update(
                        {
                            "log": path.name,
                            "rounds": t["rounds"],
                            "mode": t["timing"].get("mode"),
                            "digest": t["payload_sha256"],
                            "counts": [t["counts_white_to_move"], t["counts_black_to_move"]],
                            "changed": [r["changed_words"] for r in rounds],
                            "sweep_ms": [r.get("sweep_event_ms") for r in rounds],
                            "rounds_seconds": t["timing"].get("rounds_seconds_total"),
                            "device_bytes": t["timing"].get("device_bytes_allocated"),
                        }
                    )
            for path in sorted(self.logs.glob("verify-*.stdout")):
                for t in _json(path).get("tables", []):
                    entry = {**t, "log": path.name}
                    verifications.setdefault(t["payload_sha256"], []).append(entry)
            online_dir = self.logs.parent / "results" / "online"
            # lichess vs the local Syzygy reference files, grouped per material.
            for row in _json(online_dir / "online_comparison.json").get("rows", []):
                entry = reference_checks.setdefault(
                    row["material"],
                    {"agree": 0, "differ": 0, "unresolved": 0, "recorded_separately": 0},
                )
                key = {
                    "agree": "agree",
                    "DIFFER": "differ",
                    "unresolved": "unresolved",
                    "recorded-separately": "recorded_separately",
                }.get(row["result"], "unresolved")
                entry[key] += 1
            # lichess DTM vs a specific solved table, keyed by its payload digest.
            dtm = _json(online_dir / "online_dtm_comparison.json")
            if dtm.get("summary"):
                rows = dtm.get("rows", [])
                dtm_checks[dtm["summary"]["payload_sha256"]] = {
                    "agree": sum(r["result"] == "agree" for r in rows),
                    "compared": len(rows),
                    "metric": dtm["summary"]["metric"],
                }
        return {
            "solves": solves,
            "verifications": verifications,
            "reference_checks": reference_checks,
            "dtm_checks": dtm_checks,
        }

    def exact_dtm(self, material: str) -> bool:
        """Proof ranks equal distance to mate when every capture leads to a material
        with no decisive positions (lower children then count as rank 0 correctly)."""
        pieces = skeleton(material)
        for k in range(1, len(pieces)):
            rest = pieces[:k] + pieces[k + 1 :]
            name = "K" + "".join(_letter(r) for w, r in rest[1:] if w)
            name += "vK" + "".join(_letter(r) for w, r in rest[1:] if not w)
            solve = self.evidence["solves"].get(name)
            if solve is None or any(c["win"] or c["loss"] for c in solve["counts"]):
                return False
        return self.tables[material].rank is not None

    def table_info(self, table: Table) -> dict:
        digest = table.header["payload_sha256"]
        checks = self.evidence["verifications"].get(digest, [])
        return {
            "material": table.header["material"],
            "file": str(table.path),
            "generator": table.header.get("generator", ""),
            "digest": digest,
            "ranked": table.rank is not None,
            "exact_dtm": self.exact_dtm(table.header["material"]),
            "rounds": table.header["rounds"],
            "words": table.words,
            "verified": [
                {k: c.get(k) for k in ("status", "scope", "legal_positions", "log")} for c in checks
            ],
        }

    def materials(self) -> list[dict]:
        out = []
        for name, table in sorted(self.tables.items(), key=lambda kv: (len(kv[0]), kv[0])):
            info = self.table_info(table)
            info["solve"] = self.evidence["solves"].get(name)
            out.append(info)
        return out

    # Probing --------------------------------------------------------------------------
    def locate(self, board: chess.Board) -> tuple[Table, int, int] | None:
        """Table, word and black-king bit for a pawnless board (colours swapped if needed)."""
        for flipped in (False, True):
            table = self.tables.get(material_name(board, white=not flipped))
            if table is None:
                continue
            squares, used = [], 0
            for white, role in table.pieces:
                color = white != flipped
                free = int(board.pieces_mask(role, color)) & ~used
                square = (free & -free).bit_length() - 1
                used |= 1 << square
                squares.append(square)
            bk = board.king(chess.WHITE if flipped else chess.BLACK)
            black_to_move = (board.turn == chess.BLACK) != flipped
            return table, table.encode(black_to_move, squares), bk
        return None

    def unsupported(self, board: chess.Board) -> str | None:
        if board.pieces_mask(chess.PAWN, chess.WHITE) | board.pieces_mask(chess.PAWN, chess.BLACK):
            return "Pawns are outside the pawnless tables."
        if board.castling_rights:
            return "Castling rights are not represented; clear them."
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            return "Each side needs exactly one king."
        if (
            len(board.pieces(chess.KING, chess.WHITE)) != 1
            or len(board.pieces(chess.KING, chess.BLACK)) != 1
        ):
            return "Each side needs exactly one king."
        if chess.popcount(board.occupied) > 6:
            return "More than six pieces."
        return None

    def value(self, board: chess.Board) -> dict:
        found = self.locate(board)
        if found is None:
            return {"wdl": None, "rank": None, "table": None}
        table, word, bk = found
        wdl, rank = table.value(word, bk)
        return {"wdl": wdl, "rank": rank, "table": table.header["material"]}

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
        start = time.perf_counter()
        try:
            board = chess.Board(fen)
        except ValueError as error:
            return {"status": "invalid-fen", "message": str(error)}
        result: dict = {"fen": board.fen(), "turn": "w" if board.turn else "b"}
        reason = self.unsupported(board)
        if reason:
            return {**result, "status": "unsupported", "message": reason}
        found = self.locate(board)
        if found is None:
            name = material_name(board, white=True)
            return {**result, "status": "missing-table", "message": f"No table for {name}."}
        table, word, bk = found
        wdl, rank = table.value(word, bk)
        legal_by_chess = chess_legal(board)
        result |= {
            "material": table.header["material"],
            "table": self.table_info(table),
            "check": board.is_check(),
        }
        if wdl is None or not legal_by_chess:
            result |= {
                "status": "illegal",
                "message": "Illegal position: the side not to move is in check, or kings touch.",
                "agrees_with_python_chess": (wdl is None) == (not legal_by_chess),
            }
            return result
        syzygy = self.syzygy_wdl(board)
        moves = []
        for move in board.legal_moves:
            san = board.san(move)
            capture = board.is_capture(move)
            board.push(move)
            child = self.value(board)
            child_syzygy = self.syzygy_wdl(board) if child["wdl"] is not None else None
            moves.append(
                {
                    "fen": board.fen(),
                    "uci": move.uci(),
                    "san": san,
                    "capture": capture,
                    "checkmate": board.is_checkmate(),
                    "stalemate": board.is_stalemate(),
                    "wdl": None if child["wdl"] is None else -child["wdl"],
                    "child_rank": child["rank"],
                    "child_table": child["table"],
                    "syzygy_agrees": None
                    if child_syzygy is None or child["wdl"] is None
                    else (child_syzygy > 0) - (child_syzygy < 0) == child["wdl"] // 2,
                }
            )
            board.pop()
        moves.sort(key=move_order)
        result |= {
            "status": "ok",
            "wdl": wdl,
            "rank": rank,
            "checkmate": board.is_checkmate(),
            "stalemate": board.is_stalemate(),
            "syzygy": None
            if syzygy is None
            else {
                "wdl": syzygy,
                "agrees": (syzygy > 0) - (syzygy < 0) == wdl // 2,
            },
            "moves": moves,
        }
        result["probe_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    def random_position(self, material: str, turn: str, want: str) -> dict:
        table = self.tables.get(material)
        if table is None:
            return {"status": "missing-table", "message": f"No table for {material}."}
        rng = random.SystemRandom()
        goal = {"win": 2, "draw": 0, "loss": -2}.get(want)
        for _ in range(200_000):
            word = rng.randrange(table.words // 2)
            if turn == "b" or (turn == "any" and rng.random() < 0.5):
                word += table.words // 2
            bk = rng.randrange(64)
            wdl, _ = table.value(word, bk)
            if wdl is None or (goal is not None and wdl != goal):
                continue
            black_to_move, squares = table.decode(word)
            board = chess.Board(None)
            for (white, role), square in zip(table.pieces, squares, strict=True):
                board.set_piece_at(square, chess.Piece(role, white))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.turn = not black_to_move
            return {"status": "ok", "fen": board.fen()}
        return {"status": "not-found", "message": f"No {want} position found for {material}."}


def chess_legal(board: chess.Board) -> bool:
    """python-chess legality, accepting check patterns that cannot arise in play
    (the tables and the shakmaty verifier accept them too)."""
    ignored = chess.STATUS_IMPOSSIBLE_CHECK | chess.STATUS_TOO_MANY_CHECKERS
    return board.status() & ~ignored == chess.STATUS_VALID


def move_order(m: dict) -> tuple:
    """lichess order: wins (fastest first), draws, losses (longest first), unknown last."""
    rank = m["child_rank"] if m["child_rank"] is not None else 1 << 20
    if m["wdl"] == 2:
        return (0, 0 if m["checkmate"] else 1, rank, m["san"])
    if m["wdl"] == 0:
        return (1, 0, 0, m["san"])
    if m["wdl"] == -2:
        return (2, 0, -rank, m["san"])
    return (3, 0, 0, m["san"])


def _letter(role: int) -> str:
    return chess.piece_symbol(role).upper()


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


@functools.lru_cache(maxsize=16)
def piece_svg(code: str) -> bytes:
    color, role = code[0], code[1].upper()
    symbol = role if color == "w" else role.lower()
    return chess.svg.piece(chess.Piece.from_symbol(symbol)).encode()


class Handler(http.server.BaseHTTPRequestHandler):
    store: Store
    types: typing.ClassVar[dict[str, str]] = {
        ".html": "text/html",
        ".js": "text/javascript",
        ".css": "text/css",
        ".svg": "image/svg+xml",
        ".woff2": "font/woff2",
        ".txt": "text/plain",
    }

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def send(self, body: bytes, kind: str, status: int = 200) -> None:
        self.send_response(status)
        textual = kind.startswith(("text/", "application/json", "image/svg"))
        self.send_header("Content-Type", f"{kind}; charset=utf-8" if textual else kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if kind == "application/json" else "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        query = dict(urllib.parse.parse_qsl(url.query))
        api = {
            "/api/probe": lambda: self.store.probe(query.get("fen", "")),
            "/api/materials": self.store.materials,
            "/api/evidence": lambda: self.store.evidence,
            "/api/random": lambda: self.store.random_position(
                query.get("material", ""), query.get("turn", "any"), query.get("result", "any")
            ),
        }
        if url.path in api:
            self.send(json.dumps(api[url.path]()).encode(), "application/json")
        elif url.path.startswith("/piece/") and url.path.endswith(".svg"):
            code = url.path[len("/piece/") : -4]
            if len(code) == 2 and code[0] in "wb" and code[1] in "KQRBNP":
                self.send(piece_svg(code), "image/svg+xml")
            else:
                self.send(b"not found", "text/plain", 404)
        else:
            name = "index.html" if url.path in ("/", "") else url.path.lstrip("/")
            path = (STATIC / name).resolve()
            if not path.is_relative_to(STATIC) or not path.is_file():
                self.send(b"not found", "text/plain", 404)
                return
            self.send(path.read_bytes(), self.types.get(path.suffix, "application/octet-stream"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", type=Path, action="append", required=True)
    parser.add_argument("--syzygy", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--port", type=int, default=8642)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default loopback)")
    args = parser.parse_args()
    Handler.store = Store(args.tables, args.syzygy, args.logs)
    print(f"{len(Handler.store.tables)} tables; http://{args.host}:{args.port}/", flush=True)
    http.server.ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
