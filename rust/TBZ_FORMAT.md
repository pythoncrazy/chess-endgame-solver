# `.tbz` pawnless table format, version 4

Reference implementation: `src/packed.rs` (writer, reader, probe, decode).
Independent reader: `tools/visualizer/server.py` (`PackedTable`).
Version 3 files (`TBPAWNZ1`) remain readable; they are version 4 with both
rank sides stored, zstd blocks and no WDL code 3.

## Goals

- Smallest file that still answers a single position from at most one block
  per section (plus, for a black-to-move rank, the blocks of its children).
- Exact: `tb_pack --check` decodes every file and requires bit equality with
  the solver's table; `--probes N` also checks N single-position probes.
- Self-describing and authenticated: a JSON header, a payload digest and a
  digest of the decompressed content.

## File layout

| bytes | content |
|---|---|
| 0..8 | magic `TBPAWNZ2` |
| 8..16 | header length `L`, u64 little-endian (at most 1 MiB) |
| 16..16+L | header, UTF-8 JSON |
| 16+L.. | payload: the sections, at the offsets the header gives |

Header fields: `format` (`tablebase-pawnless-v4`), `material` (for example
`KQBvKR`), `layout` (prose summary), `stored_words`, `rounds`, `rank_width`
(0, 1 or 2 bytes), `block_bytes` (2^18), `zstd_level`, `sections`,
`rank_sides` (`both` or `white`), `counts` (legal/win/loss totals per side to
move), `payload_sha256`, `digest_scheme`, `content_sha256`, `generator`.
Readers must ignore unknown fields; missing `rank_sides` means `both` and a
missing section `codec` means `zstd`.

Each section has `name`, `offset` (from the payload start), `blocks`,
`raw_bytes` (decompressed), `compressed_bytes` and `codec`. At `offset` lies
an index of `blocks + 1` u64 offsets relative to the first frame; frame `b`
spans `[index[b], index[b+1])` and decompresses to `block_bytes` bytes
(the last block may be shorter). Codecs:

- `zstd`: one zstd frame per block (level 19).
- `lzma`: one LZMA1 "alone" (`.lzma`) stream per block, preset 9e with
  `lc=4`, dictionary = block size. Python: `lzma.FORMAT_ALONE`.

## Stored domain and index

The skeleton is every piece except the black king, in material order: white
king, white pieces, black pieces. A position is stored once per symmetry
class of the eight board symmetries:

1. Map the white king into the triangle a1-d1-d4 (`TRIANGLE` =
   a1 b1 c1 d1 b2 c2 d2 c3 d3 d4).
2. If the white king is on the a1-h8 diagonal, the transpose also keeps it
   there: take the image with the smaller skeleton index; if the skeleton is
   transpose-invariant, put the black king on or below the diagonal
   (`LOWER_RIGHT`).
3. Identical pieces take ascending squares.

Stored index: `p = ((turn * 10 + triangle(white king)) * 64^(m-1) + other
skeleton squares in base 64) * 64 + black king`, with `turn` 0 for white to
move and `m` the skeleton length. White-to-move positions are `p < 10 *
64^m`.

Legality is not stored: readers derive it (both kings, no overlap, the side
not to move not in check). Illegal positions and indices that are not the
canonical image are "don't care" and repeat the previous stored value, which
compresses to almost nothing.

## Sections

`wdl` (always): 2 bits per stored index, four positions per byte, low bits
first. 0 draw, 1 win, 2 loss for the side to move, 3 a black-to-move win of
rank 1 (only when `rank_sides` is `white`).

`rank` (when `rank_width > 0`): proof rank of decisive positions, u8 or u16
little-endian per stored index. With `rank_sides = both` it covers every
stored index; with `white` it covers only white-to-move indices.

Proof rank: checkmate is 0, a win is one more than its fastest losing
child, and a loss is one more than its slowest child. Captures lead to lower
material, which counts as rank 0.

## Black-to-move ranks (`rank_sides = white`)

A decisive black-to-move position with code 3 has rank 1. Otherwise its rank
follows from its legal moves, where every non-capture leads to a
white-to-move position of the same material whose rank is stored:

- win: 1 + min rank over non-capture children that are losses (a capture
  that wins would have given rank 1, code 3);
- loss: 0 without legal moves (checkmate), else 1 + max rank over
  non-capture children (captures count as 0).

Dropping black ranks removes about 60% of the file (KQBvKR: 37.9 to 15.7 MB)
at the cost of reading the children when a black-to-move rank is asked for.

## Integrity

`payload_sha256` under `digest_scheme = sha256-tree-16MiB`: SHA-256 of the
SHA-256 digests of consecutive 16 MiB payload chunks, followed by the payload
length as u64 little-endian. `content_sha256`: SHA-256 of the SHA-256
digests of every decompressed block, sections in header order, which
identifies the content independently of codec and level.
