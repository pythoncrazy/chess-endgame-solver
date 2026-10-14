"""Print material and payload SHA-256 recorded in pawnless table files."""

import json
import sys
from pathlib import Path


class NotATableError(ValueError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"{path}: not a pawnless table file")


def header(path: Path) -> dict:
    with path.open("rb") as f:
        if f.read(8) != b"TBPAWNL1":
            raise NotATableError(path)
        length = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(length))


for name in sys.argv[1:]:
    h = header(Path(name))
    print(h["material"], h["payload_sha256"], h["rounds"], Path(name).name)
