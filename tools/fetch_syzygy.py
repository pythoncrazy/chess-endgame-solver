"""Fetch the published Syzygy WDL tables of every pawnless material up to five pieces.

Files come from tablebase.lichess.ovh. Each file must match the MD5 that an
independent mirror (tablebase.sesse.net) publishes. The catalog records size, MD5
and SHA-256. This script downloads only reference data. It generates no chess result.

Usage: python -I tools/fetch_syzygy.py DIRECTORY [MAX_PIECES]
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

SOURCE = "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/"
CHECKSUMS = "http://tablebase.sesse.net/syzygy/3-4-5/checksum.md5"
VALUE = {"Q": 9, "R": 5, "B": 3, "N": 3}


def syzygy_name(white: str, black: str) -> str:
    """Syzygy file name: the side with more pieces (then more value) first, QRBN order."""
    order = "QRBN"
    w = "".join(sorted(white, key=order.index))
    b = "".join(sorted(black, key=order.index))
    key = lambda s: (len(s), sum(VALUE[c] for c in s), [-order.index(c) for c in s])  # noqa: E731
    if key(b) > key(w):
        w, b = b, w
    return f"K{w}vK{b}"


def materials(max_pieces: int) -> list[str]:
    names = set()
    for total in range(1, max_pieces - 1):
        for combo in itertools.combinations_with_replacement("QRBN", total):
            for split in range(total + 1):
                for white in itertools.combinations(range(total), split):
                    w = "".join(combo[i] for i in white)
                    b = "".join(combo[i] for i in range(total) if i not in white)
                    names.add(syzygy_name(w, b))
    return sorted(names, key=lambda n: (len(n), n))


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - fixed https/http hosts
        return response.read()


def main() -> int:
    root = Path(sys.argv[1])
    max_pieces = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    root.mkdir(parents=True, exist_ok=True)
    md5 = {}
    for line in fetch(CHECKSUMS).decode().splitlines():
        digest, _, name = line.partition("  ")
        md5[name.strip()] = digest
    catalog = []
    for name in materials(max_pieces):
        file = f"{name}.rtbw"
        expected = md5.get(file)
        if expected is None:
            print(f"{file}: not in the independent checksum list", file=sys.stderr)
            return 1
        path = root / file
        data = path.read_bytes() if path.exists() else b""
        if hashlib.md5(data).hexdigest() != expected:  # noqa: S324 - published checksum format
            data = fetch(SOURCE + file)
            if hashlib.md5(data).hexdigest() != expected:  # noqa: S324
                print(f"{file}: MD5 mismatch against {CHECKSUMS}", file=sys.stderr)
                return 1
            with tempfile.NamedTemporaryFile(dir=root, delete=False) as out:
                out.write(data)
                out.flush()
                os.fsync(out.fileno())
            Path(out.name).replace(path)
        catalog.append(
            {
                "name": file,
                "bytes": len(data),
                "md5": expected,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (root / "catalog.json").write_text(
        json.dumps({"source": SOURCE, "md5_list": CHECKSUMS, "files": catalog}, indent=1) + "\n"
    )
    print(f"{len(catalog)} files, {sum(c['bytes'] for c in catalog)} bytes in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
