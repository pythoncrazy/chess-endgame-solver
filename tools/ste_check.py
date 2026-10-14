"""Check the prose of the repository against a subset of ASD-STE100 Issue 9.

The tool finds prose in Markdown, text, HTML and code comments. It checks the
rules that a program can check:

- A sentence has a maximum of 20 words (rules 5.1 and 6.1 let you use 20 and 25).
- Prose has no semicolons and no contractions.
- Prose does not use Latin abbreviations.
- Words that end in "ing" are technical names from TERMS.
- Words from UNAPPROVED do not occur. The tool shows the approved word.

Approved meanings, parts of speech and noun clusters must have a human review.
This tool does not certify compliance.

Usage: python tools/ste_check.py [PATH...]
"""

from __future__ import annotations

import ast
import html.parser
import io
import re
import shutil
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

MAX_WORDS = 20

# Unapproved words and the approved words to use instead.
UNAPPROVED = {
    "achieve": "get",
    "additional": "more",
    "additionally": "also",
    "allows": "lets",
    "approx": "approximately",
    "assist": "help",
    "attempt": "try",
    "begin": "start",
    "begins": "starts",
    "capable": "can",
    "commence": "start",
    "comprise": "contain",
    "comprises": "contains",
    "currently": "now",
    "determine": "find",
    "determines": "finds",
    "eliminate": "remove",
    "eliminates": "removes",
    "enable": "let",
    "enables": "lets",
    "ensure": "make sure",
    "ensures": "makes sure",
    "entire": "all",
    "establish": "make",
    "however": "but",
    "huge": "very large",
    "indicate": "show",
    "indicates": "shows",
    "initial": "first",
    "initially": "first",
    "initiate": "start",
    "lots": "many",
    "maintain": "keep",
    "maintains": "keeps",
    "modify": "change",
    "modifies": "changes",
    "multiple": "many or several",
    "need": "necessary or must",
    "needed": "necessary",
    "needs": "must",
    "obtain": "get",
    "obtains": "gets",
    "perform": "do",
    "performs": "does",
    "performed": "done",
    "permit": "let",
    "possess": "have",
    "prior": "before",
    "proceed": "continue",
    "provide": "give",
    "provides": "gives",
    "provided": "given",
    "require": "must or necessary",
    "requires": "must or necessary",
    "required": "necessary",
    "retain": "keep",
    "retains": "keeps",
    "sufficient": "enough",
    "terminate": "stop",
    "utilize": "use",
    "various": "different",
    "via": "through",
    "whether": "if",
    "within": "in",
}
# Technical names and approved words that end in "ing".
ING_WORDS = {
    "anything",
    "bring",
    "castling",
    "ceiling",
    "during",
    "everything",
    "king",
    "kings",
    "nothing",
    "padding",
    "ring",
    "something",
    "spring",
    "string",
    "strings",
    "thing",
    "things",
    "working",
    "zeroing",
}
CONTRACTION = re.compile(r"\b\w+(n't|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE)
LATIN = re.compile(r"\b(e\.g\.|i\.e\.|etc\.|vs\.|viz\.|cf\.)", re.IGNORECASE)
CODE_SPAN = re.compile(r"`[^`]*`")
URL = re.compile(r"https?://\S+|\b[\w.-]+\.(?:rs|py|md|json|toml|sh|js|html|css|tbz|tbp)\b")
# Identifiers, paths, numbers with units and formulas are not prose words.
NOT_WORD = re.compile(r"[_/\\=<>{}\[\]()|^*+~#$@0-9]|::|[a-z][A-Z]|^[A-Z]{2,}s?$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Za-z0-9`\"(])")
SUFFIXES = {
    ".md": "markdown",
    ".txt": "text",
    ".html": "html",
    ".rs": "slash",
    ".js": "slash",
    ".mjs": "slash",
    ".css": "block",
    ".py": "python",
    ".sh": "hash",
    ".toml": "hash",
    ".yaml": "hash",
    ".yml": "hash",
}


@dataclass
class Unit:
    path: str
    line: int
    text: str


@dataclass
class Problem:
    unit: Unit
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.unit.path}:{self.unit.line}: {self.rule}: {self.detail}"


def markdown_units(path: str, text: str) -> list[Unit]:
    units: list[Unit] = []
    fence = False
    block: list[str] = []
    start = 0
    for number, raw in enumerate([*text.splitlines(), ""], 1):
        line = raw.strip()
        if line.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        item = re.match(r"^([-*]|\d+\.)\s+", line)
        heading = line.startswith("#")
        table = line.startswith("|")
        if (not line or item or heading or table) and block:
            units.append(Unit(path, start, " ".join(block)))
            block = []
        if table:
            cells = [c.strip() for c in line.strip("|").split("|")]
            units.extend(Unit(path, number, c) for c in cells if c and not set(c) <= set("-: "))
            continue
        if heading:
            units.append(Unit(path, number, line.lstrip("#").strip()))
            continue
        if line:
            if not block:
                start = number
            block.append(line[item.end() :] if item else line)
    return units


def text_units(path: str, text: str) -> list[Unit]:
    if re.search(r"^```", text, re.MULTILINE):
        return markdown_units(path, text)
    units: list[Unit] = []
    block: list[str] = []
    start = 0
    for number, raw in enumerate([*text.splitlines(), ""], 1):
        line = raw.strip()
        code = raw.startswith("    ") or raw.startswith("\t")
        if not line or code or re.match(r"^([-*]|\d+\.)\s+", line):
            if block:
                units.append(Unit(path, start, " ".join(block)))
                block = []
            if code or not line:
                continue
            line = re.sub(r"^([-*]|\d+\.)\s+", "", line)
        if not block:
            start = number
        block.append(line)
    return units


class _Html(html.parser.HTMLParser):
    SKIP = frozenset({"script", "style", "code", "pre", "svg"})

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.units: list[Unit] = []
        self.depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.depth += 1
        for name, value in attrs:
            if name in {"title", "aria-label", "placeholder", "alt"} and value:
                self.units.append(Unit(self.path, self.getpos()[0], value))

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self.depth:
            self.depth -= 1

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if text and not self.depth:
            self.units.append(Unit(self.path, self.getpos()[0], text))


def html_units(path: str, text: str) -> list[Unit]:
    parser = _Html(path)
    parser.feed(text)
    return parser.units


def comment_blocks(path: str, lines: list[tuple[int, str]]) -> list[Unit]:
    """Join comment lines that follow each other into one prose unit."""
    units: list[Unit] = []
    block: list[str] = []
    start = previous = 0
    for number, line in [*lines, (10**9, "")]:
        text = line.strip()
        if block and (number != previous + 1 or not text or re.match(r"^([-*]|\d+\.)\s", text)):
            units.append(Unit(path, start, " ".join(block)))
            block = []
        if text:
            if not block:
                start = number
            block.append(re.sub(r"^([-*]|\d+\.)\s+", "", text))
        previous = number
    return units


def slash_units(path: str, text: str) -> list[Unit]:
    lines = []
    for number, raw in enumerate(text.splitlines(), 1):
        match = re.match(r"^\s*//[/!]?\s?(.*)$", raw) or re.search(r"\s//\s(.*)$", raw)
        if match:
            lines.append((number, match.group(1)))
    return comment_blocks(path, lines)


def block_units(path: str, text: str) -> list[Unit]:
    units = []
    for match in re.finditer(r"/\*(.*?)\*/", text, re.DOTALL):
        line = text.count("\n", 0, match.start()) + 1
        units.append(Unit(path, line, " ".join(match.group(1).split())))
    return units


def hash_units(path: str, text: str) -> list[Unit]:
    lines = []
    for number, raw in enumerate(text.splitlines(), 1):
        if raw.startswith("#!"):
            continue
        match = re.match(r"^\s*#\s?(.*)$", raw) or re.search(r"\s#\s(.*)$", raw)
        if match:
            lines.append((number, match.group(1)))
    return comment_blocks(path, lines)


def python_units(path: str, text: str) -> list[Unit]:
    lines = [
        (token.start[0], token.string.lstrip("#").strip())
        for token in tokenize.generate_tokens(io.StringIO(text).readline)
        if token.type == tokenize.COMMENT and not token.string.startswith("#!")
    ]
    units = comment_blocks(path, lines)
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            doc = ast.get_docstring(node)
            if doc:
                line = node.body[0].lineno
                units.extend(Unit(u.path, u.line + line - 1, u.text) for u in text_units(path, doc))
    return units


EXTRACT = {
    "markdown": markdown_units,
    "text": text_units,
    "html": html_units,
    "slash": slash_units,
    "block": block_units,
    "python": python_units,
    "hash": hash_units,
}


def words(sentence: str) -> list[str]:
    sentence = URL.sub(" x ", CODE_SPAN.sub(" x ", sentence))
    return [w.strip(".,:!?\"'()") for w in sentence.split() if w.strip(".,:!?\"'()-")]


def check(unit: Unit) -> list[Problem]:
    problems = []
    prose = CODE_SPAN.sub(" ", unit.text)
    if ";" in URL.sub(" ", prose):
        problems.append(Problem(unit, "semicolon", "use two sentences or a list"))
    for match in CONTRACTION.finditer(prose):
        problems.append(Problem(unit, "contraction", match.group(0)))
    for match in LATIN.finditer(prose):
        problems.append(Problem(unit, "abbreviation", match.group(0)))
    for sentence in SENTENCE_END.split(unit.text):
        count = len(words(sentence))
        if count > MAX_WORDS:
            problems.append(Problem(unit, "length", f"{count} words: {sentence[:60]}..."))
    for word in (part for w in words(prose) for part in w.split("-")):
        if not word or NOT_WORD.search(word) or word.isupper():
            continue
        lower = word.lower()
        if lower in UNAPPROVED:
            problems.append(Problem(unit, "word", f"{word} -> {UNAPPROVED[lower]}"))
        elif lower.endswith("ing") and lower not in ING_WORDS and len(lower) > 4:
            problems.append(Problem(unit, "ing", word))
    return problems


def tracked(paths: list[str]) -> list[Path]:
    git = shutil.which("git") or "git"
    out = subprocess.run(  # noqa: S603 (the arguments are paths for git ls-files)
        [git, "ls-files", "--", *paths], capture_output=True, text=True, check=True
    ).stdout
    return [Path(p) for p in out.splitlines() if Path(p).suffix in SUFFIXES]


def main(argv: list[str]) -> int:
    problems: list[Problem] = []
    count = 0
    for path in tracked(argv or ["."]):
        licence = path.name.startswith(("OFL", "LICENSE"))
        if licence or path.name == "Cargo.lock" or "reference_directory_listings" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for unit in EXTRACT[SUFFIXES[path.suffix]](str(path), text):
            count += 1
            problems.extend(check(unit))
    for problem in problems:
        print(problem)
    print(f"{count} prose units, {len(problems)} problems", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
