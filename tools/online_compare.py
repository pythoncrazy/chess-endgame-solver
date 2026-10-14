"""Compare local WDL results with tablebase.lichess.ovh (rate-limited, cached).

Usage: python -I tools/online_compare.py PACKAGE_ROOT REFERENCE_DIR OUTPUT_DIR

The local oracle is python-chess probing the verified native Syzygy files. For the
1,005-FEN set, the Rust-verified expected values must also agree. Unknown or
unmapped online categories are recorded as unresolved, never as draws.
"""

from __future__ import annotations

import collections
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import chess
import chess.syzygy

SERVICE = "https://tablebase.lichess.ovh/standard"
CATEGORY = {"win": 2, "cursed-win": 1, "draw": 0, "blessed-loss": -1, "loss": -2}
MIN_INTERVAL_SECONDS = 1.1
SAMPLE_SIZE = 64
SEED = 20261014

EXTRA_POSITIONS = [
    ("reference-check", "8/8/8/8/8/7k/4r3/KQ6 w - - 0 1"),
    ("checkmate-KQvK", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
    ("stalemate-KQvK", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
    ("checkmate-KRvK", "R6k/8/6K1/8/8/8/8/8 b - - 0 1"),
    ("capture-KQvKR-white", "8/8/8/8/8/2k5/1r6/KQ6 w - - 0 1"),
    ("capture-KQvKR-black", "8/8/8/8/8/2k5/1r6/KQ6 b - - 0 1"),
    ("KQvK-white", "8/8/8/4k3/8/8/8/KQ6 w - - 0 1"),
    ("KQvK-black", "8/8/8/4k3/8/8/8/KQ6 b - - 0 1"),
    ("KRvK-black", "8/8/3k4/8/8/8/8/R3K3 b - - 0 1"),
    ("KNNvK-draw", "8/8/3k4/8/8/8/8/1NN1K3 w - - 0 1"),
    ("double-check-KRBvK", "4k3/8/8/1B6/8/8/8/K3R3 b - - 0 1"),
    ("KNNvKNN-repeated", "1n2k1n1/8/8/8/8/8/8/1N2K1N1 w - - 0 1"),
]
CLOCKED_POSITIONS = [
    ("clock-60-cursed-check", "6N1/1kp5/8/8/8/7N/8/5K2 w - - 60 1"),
    ("clock-60-KQvKR", "8/8/8/8/8/7k/4r3/KQ6 w - - 60 1"),
]


def material(board: chess.Board) -> str:
    """Return a Syzygy-style material name such as KQvKR."""

    def side(color: chess.Color) -> str:
        return "".join(
            chess.piece_symbol(piece).upper() * len(board.pieces(piece, color))
            for piece in (
                chess.KING,
                chess.QUEEN,
                chess.ROOK,
                chess.BISHOP,
                chess.KNIGHT,
                chess.PAWN,
            )
        )

    return f"{side(chess.WHITE)}v{side(chess.BLACK)}"


def probe_clock_zero(tablebase: chess.syzygy.Tablebase, board: chess.Board) -> int | None:
    """WDL after zeroing: probe with the halfmove clock set to zero."""
    zero = board.copy()
    zero.halfmove_clock = 0
    try:
        return tablebase.probe_wdl(zero)
    except (KeyError, chess.syzygy.MissingTableError):
        return None


class Client:
    """Rate-limited HTTP client with an append-only JSONL response log/cache."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.cache: dict[str, dict] = {}
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                record = json.loads(line)
                if record["status"] == 200:
                    self.cache[record["fen"]] = record
        self.interval = MIN_INTERVAL_SECONDS
        self.last_request = 0.0
        self.requests_sent = 0

    def _request_once(self, url: str) -> tuple[int | None, str, str | None]:
        request = urllib.request.Request(  # noqa: S310 - fixed https service URL
            url, headers={"User-Agent": "tablebase-lab-verification/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return response.status, response.read().decode(), None
        except urllib.error.HTTPError as error:
            body = error.read().decode(errors="replace")
            return error.code, body, error.headers.get("Retry-After")
        except (urllib.error.URLError, TimeoutError) as error:
            return None, repr(error), None

    def get(self, fen: str) -> tuple[dict, bool]:
        """Return (record, from_cache). Obeys the interval, 429 and Retry-After."""
        if fen in self.cache:
            return self.cache[fen], True
        url = f"{SERVICE}?{urllib.parse.urlencode({'fen': fen})}"
        record: dict = {}
        for attempt in range(8):
            wait = self.last_request + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last_request = time.monotonic()
            sent = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            status, body, retry_after = self._request_once(url)
            self.requests_sent += 1
            record = {"fen": fen, "url": url, "time_utc": sent, "status": status, "body": body}
            with self.log_path.open("a") as log:
                log.write(json.dumps(record) + "\n")
            if status == 200:
                self.cache[fen] = record
                return record, False
            if status == 429:
                self.interval *= 2
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 60)
            else:
                time.sleep(5 * (attempt + 1))
        return record, False


def select_positions(rust_values: dict[str, int]) -> list[tuple[str, str]]:
    """Stratify by material, value and turn; then fill randomly to SAMPLE_SIZE."""
    rng = random.Random(SEED)  # noqa: S311 - reproducible sampling, not cryptography
    groups: dict[tuple[str, int, str], list[str]] = collections.defaultdict(list)
    for fen, value in rust_values.items():
        groups[material(chess.Board(fen)), value, fen.split()[1]].append(fen)
    selected = [("six-piece-stratified", rng.choice(groups[key])) for key in sorted(groups)]
    chosen = {fen for _, fen in selected}
    rest = [fen for fen in rust_values if fen not in chosen]
    rng.shuffle(rest)
    while len(selected) < SAMPLE_SIZE:
        selected.append(("six-piece-random", rest.pop()))
    return selected


def compare_children(tablebase, board: chess.Board, moves: list[dict]) -> dict:
    """Compare each online child entry with the local child value at clock zero.

    Online child categories include the child's own clock. A quiet child with a
    nonzero clock whose class magnitude differs is recorded as skipped.
    """
    differ, checked, skipped = [], 0, 0
    for move in moves:
        child = board.copy()
        child.push_uci(move["uci"])
        local = probe_clock_zero(tablebase, child)
        if move["category"] not in CATEGORY or local is None:
            skipped += 1
            continue
        online = CATEGORY[move["category"]]
        clock_dependent = not move["zeroing"] and child.halfmove_clock != 0
        if clock_dependent and abs(online) != abs(local):
            skipped += 1
            continue
        checked += 1
        if online != local:
            differ.append({"uci": move["uci"], "online": move["category"], "local_clock0": local})
    return {"children_checked": checked, "children_skipped": skipped, "children_differ": differ}


def compare_position(tablebase, client: Client, label: str, fen: str, rust: int | None) -> dict:
    board = chess.Board(fen)
    local = probe_clock_zero(tablebase, board)
    record, cached = client.get(fen)
    row = {
        "label": label,
        "fen": fen,
        "turn": fen.split()[1],
        "halfmove_clock": board.halfmove_clock,
        "material": material(board),
        "local_wdl_python_chess": local,
        "local_wdl_rust_verified": rust,
        "http_status": record["status"],
        "response_time_utc": record["time_utc"],
        "source_url": record["url"],
        "cached": cached,
    }
    if record["status"] != 200:
        return {**row, "result": "unresolved", "reason": "no HTTP 200 response"}
    data = json.loads(record["body"])
    category = data.get("category")
    online_moves = data.get("moves", [])
    row |= {
        "online_category": category,
        "online_dtz": data.get("dtz"),
        "online_precise_dtz": data.get("precise_dtz"),
        "online_dtm": data.get("dtm"),
        "legal_moves_local": board.legal_moves.count(),
        "legal_move_sets_equal": {m["uci"] for m in online_moves}
        == {m.uci() for m in board.legal_moves},
        "terminal_flags_equal": data.get("checkmate") == board.is_checkmate()
        and data.get("stalemate") == board.is_stalemate(),
        **compare_children(tablebase, board, online_moves),
    }
    if category not in CATEGORY:
        return {**row, "result": "unresolved", "reason": f"category {category!r} is not mapped"}
    if local is None:
        return {**row, "result": "unresolved", "reason": "no local reference value"}
    if board.halfmove_clock != 0:
        return {**row, "result": "recorded-separately", "clock0_wdl": local}
    agree = (
        CATEGORY[category] == local
        and rust in (None, local)
        and row["legal_move_sets_equal"]
        and row["terminal_flags_equal"]
        and not row["children_differ"]
    )
    return {**row, "result": "agree" if agree else "DIFFER"}


def main() -> int:
    package, references, output = map(Path, sys.argv[1:4])
    output.mkdir(parents=True, exist_ok=True)
    queries = json.loads((package / "six_extended_probe_queries.json").read_text())
    rust_values = dict(zip(queries["fens"], queries["values"], strict=True))
    positions = select_positions(rust_values) + EXTRA_POSITIONS + CLOCKED_POSITIONS
    tablebase = chess.syzygy.open_tablebase(str(references))
    client = Client(output / "lichess_responses.jsonl")
    rows = []
    for label, fen in positions:
        row = compare_position(tablebase, client, label, fen, rust_values.get(fen))
        rows.append(row)
        print(f"{row['result']:>20} {label:24} {fen:45} local={row['local_wdl_python_chess']}")
    counts = collections.Counter(row["result"] for row in rows)
    summary = {
        "service": SERVICE,
        "requests_sent_this_run": client.requests_sent,
        "counts": dict(counts),
        "distinct_clock0_positions": len({r["fen"] for r in rows if r["halfmove_clock"] == 0}),
        "wdl_classes_seen": sorted(
            {r["online_category"] for r in rows if r.get("online_category")}
        ),
        "materials": sorted({r["material"] for r in rows}),
        "mapping": CATEGORY,
        "dtz_compared": False,
        "dtz_reason": "local DTZ (.rtbz) references are not downloaded; online DTZ is recorded",
    }
    report = json.dumps({"summary": summary, "rows": rows}, indent=2)
    (output / "online_comparison.json").write_text(report + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if counts["DIFFER"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
