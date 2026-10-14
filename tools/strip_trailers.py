"""Remove co-author and tool trailers from commit messages.

The repository has one author. A commit message must not have co-author or tool lines.

Usage:
    uv run tools/strip_trailers.py MESSAGE_FILE    (commit-msg hook: edit the file)
    uv run tools/strip_trailers.py --check REV_RANGE    (CI: fail if a commit has one)
    uv run tools/strip_trailers.py --check-text FILE    (CI: fail if a text has one)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TRAILER = re.compile(
    r"^\s*(co-authored-by|made-with|generated-by|assisted-by)\s*:.*$"
    r"|^\s*(made|generated) with \[[^\]]*\]\(https?://[^)]*\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip(text: str) -> str:
    """The text without the trailer lines and without blank lines at the end."""
    lines = [line for line in text.splitlines() if not TRAILER.fullmatch(line)]
    return "\n".join(lines).rstrip() + "\n"


def check_range(revisions: str) -> int:
    """Count the commits in the range that have a trailer line."""
    log = subprocess.run(  # noqa: S603 - fixed argument list, no shell
        ["git", "log", "--format=%H%x00%B%x01", revisions],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    bad = 0
    for entry in filter(None, (e.strip() for e in log.split("\x01"))):
        commit, _, message = entry.partition("\x00")
        if TRAILER.search(message):
            print(f"{commit}: the message has a co-author or tool line")
            bad += 1
    return bad


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--check":
        return 1 if check_range(argv[1]) else 0
    if len(argv) == 2 and argv[0] == "--check-text":
        if TRAILER.search(Path(argv[1]).read_text()):
            print(f"{argv[1]}: the text has a co-author or tool line")
            return 1
        return 0
    if len(argv) == 1:
        path = Path(argv[0])
        text = path.read_text()
        if TRAILER.search(text):
            path.write_text(strip(text))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
