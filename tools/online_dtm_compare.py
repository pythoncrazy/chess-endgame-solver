"""Compare GPU-solved KQvK values and proof ranks with tablebase.lichess.ovh.

Usage: uv run tools/online_dtm_compare.py TABLE_FILE OUTPUT_DIR [SAMPLES]

In KQvK no capture leads to a win or loss. If Black captures the queen, the
result is a draw. Thus the synchronous proof rank of a decided position is its
exact distance to mate in plies. The lichess service reports DTM in plies, with
the sign of the side to move. Thus |dtm| must equal the rank, and the category must
match WDL.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from online_compare import CATEGORY, Client

UNRESOLVED = 0xFFFF


class TableError(ValueError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"{path}: expected a ranked KQvK pawnless table file")


def load(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = path.read_bytes()
    if raw[:8] != b"TBPAWNL1":
        raise TableError(path)
    length = int.from_bytes(raw[8:16], "little")
    header = json.loads(raw[16 : 16 + length])
    if header["material"] != "KQvK" or len(header["planes"]) != 4:
        raise TableError(path)
    words = header["words"]
    body = raw[16 + length :]
    planes = np.frombuffer(body[: 24 * words], dtype="<u8").reshape(3, words)
    rank = np.frombuffer(body[24 * words :], dtype="<u2")
    return header, planes[0], planes[1], planes[2], rank


def fen_of(word: int, black_king: int) -> str:
    black_to_move, white_king, queen = word >> 12, (word >> 6) & 63, word & 63
    board = chess.Board(None)
    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(queen, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))
    board.turn = not black_to_move
    return board.fen()


def main() -> int:
    table, output = Path(sys.argv[1]), Path(sys.argv[2])
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    header, valid, win, loss, rank = load(table)
    rng = random.Random(20261014)  # noqa: S311 - reproducible samples
    legal = [(w, b) for w in range(len(valid)) for b in range(64) if valid[w] >> b & 1]
    picks = rng.sample(legal, samples)
    # Include the longest proof rank, so the comparison covers the DTM extreme.
    deepest = max(legal, key=lambda p: int(rank[p[0] * 64 + p[1]]) % UNRESOLVED)
    if deepest not in picks:
        picks.append(deepest)
    client = Client(output / "lichess_responses.jsonl")
    rows, agree = [], 0
    for word, bk in picks:
        fen = fen_of(word, bk)
        board = chess.Board(fen)
        value = 2 if win[word] >> bk & 1 else -2 if loss[word] >> bk & 1 else 0
        r = int(rank[word * 64 + bk])
        record, _ = client.get(fen)
        data = json.loads(record["body"]) if record["status"] == 200 else {}
        category, dtm = data.get("category"), data.get("dtm")
        wdl_ok = CATEGORY.get(category) is not None and CATEGORY[category] == value
        # lichess reports dtm 0 (or null) for draws. Decided positions must have |dtm| == rank.
        if r == UNRESOLVED:
            dtm_ok = category == "draw" and dtm in (None, 0)
        else:
            dtm_ok = dtm is not None and abs(dtm) == r
        ok = wdl_ok and dtm_ok and board.is_valid()
        agree += ok
        rows.append(
            {
                "fen": fen,
                "gpu_wdl": value,
                "gpu_rank_plies": None if r == UNRESOLVED else r,
                "online_category": category,
                "online_dtm": dtm,
                "http_status": record["status"],
                "source_url": record["url"],
                "time_utc": record["time_utc"],
                "result": "agree" if ok else "DIFFER",
            }
        )
        print(f"{rows[-1]['result']:>6} {fen:40} wdl={value} rank={r} online={category} dtm={dtm}")
    summary = {
        "table": str(table),
        "payload_sha256": header["payload_sha256"],
        "positions": len(rows),
        "agree": agree,
        "metric": "WDL category and |DTM| plies vs synchronous proof rank",
        "draw_dtm_interpretation": "online dtm 0 or null with category draw",
    }
    report = json.dumps({"summary": summary, "rows": rows}, indent=2)
    (output / "online_dtm_comparison.json").write_text(report + "\n")
    print(json.dumps(summary))
    return 0 if agree == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
