# `.tbz` pawnless table format, version 4

Reference implementation: `src/packed.rs` (writer, reader, probe, decode).
Independent reader: `tools/visualizer/server.py` (`PackedTable`).
Readers can still read version 3 files (`TBPAWNZ1`). Version 3 is version 4
with both rank sides stored, zstd blocks and no WDL code 3.

## Goals

- Smallest file that answers a single position from a maximum of one block in
  each section. For a black-to-move rank, it also reads the blocks of the
  children.
- Exact: `tb_pack --check` decodes every file and makes sure that it is
  equal, bit for bit, to the solver table. `--probes N` also checks N
  single-position probes.
- Contains its own description and authentication: a JSON header, a payload
  digest and a digest of the decompressed content.

## File layout

| bytes | content |
|---|---|
| 0..8 | magic `TBPAWNZ2` |
| 8..16 | header length `L`, u64 little-endian (at most 1 MiB) |
| 16..16+L | header, UTF-8 JSON |
| 16+L.. | payload: the sections, at the offsets the header gives |

The header has these fields: `format` (`tablebase-pawnless-v4`), `material`
(for example `KQBvKR`) and `layout` (prose summary). It also has
`stored_words`, `rounds`, `rank_width` (0, 1 or 2 bytes), `block_bytes` (2^18),
`zstd_level` and `sections`. The other fields are `rank_sides` (`both` or
`white`), `counts` (legal/win/loss totals per side to move), `payload_sha256`,
`digest_scheme`, `content_sha256` and `generator`. Readers must ignore unknown
fields. If `rank_sides` is absent, its value is `both`. If a section has no
`codec`, its codec is `zstd`.

Each section has `name`, `offset` (from the payload start), `blocks`,
`raw_bytes` (decompressed), `compressed_bytes` and `codec`. At `offset` there
is an index of `blocks + 1` u64 offsets, relative to the first frame. Frame `b`
spans `[index[b], index[b+1])`. It decompresses to `block_bytes` bytes, but the
last block can be shorter. Codecs:

- `zstd`: one zstd frame per block (level 19).
- `lzma`: one LZMA1 "alone" (`.lzma`) stream per block, preset 9e with
  `lc=4`, dictionary = block size. Python: `lzma.FORMAT_ALONE`.
- `cm`: a context model with a binary arithmetic coder. A `cm` section has a
  different layout. [TBZ_CM.md](TBZ_CM.md) gives it.

## Stored domain and index

The skeleton is every piece except the black king, in material order: white
king, white pieces, black pieces. The file stores a position one time for each
symmetry class of the eight board symmetries:

1. Map the white king into the triangle a1-d1-d4 (`TRIANGLE` =
   a1 b1 c1 d1 b2 c2 d2 c3 d3 d4).
2. If the white king is on the a1-h8 diagonal, the transpose also keeps it
   there. Then take the image with the smaller skeleton index. If the transpose
   does not change the skeleton, put the black king on or below the diagonal
   (`LOWER_RIGHT`).
3. Identical pieces take their squares from low to high.

Stored index: `p = ((turn * 10 + triangle(white king)) * 64^(m-1) + other
skeleton squares in base 64) * 64 + black king`, with `turn` 0 for white to
move and `m` the skeleton length. White-to-move positions are `p < 10 *
64^m`.

The file does not store legality. Readers derive it: both kings, no overlap,
and the side not to move is not in check. Illegal positions and indices that
are not the canonical image are "do not care" entries. They repeat the previous
stored value, which compresses to almost nothing.

## Sections

`wdl` (always): 2 bits per stored index, four positions per byte, low bits
first. Codes 0 draw, 1 win and 2 loss are for the side to move. Code 3 is a
black-to-move win of rank 1 (only when `rank_sides` is `white`).

`rank` (when `rank_width > 0`): proof rank of decisive positions, u8 or u16
little-endian per stored index. With `rank_sides = both`, it covers every
stored index. With `white`, it covers only white-to-move indices.

Proof rank: checkmate is 0. A win is one more than the lowest rank of a child
that is a loss. A loss is one more than the highest rank of its children.
Captures lead to lower material, which counts as rank 0.

## Black-to-move ranks (`rank_sides = white`)

A decisive black-to-move position with code 3 has rank 1. Otherwise, its legal
moves give its rank. Every non-capture leads to a white-to-move position of the
same material with a stored rank:

- win: 1 + min rank over non-capture children that are losses. If a capture
  wins, the position has rank 1 and code 3.
- loss: 0 without legal moves (checkmate), else 1 + max rank over
  non-capture children (captures count as 0).

Without black ranks, the file is approximately 60% smaller (KQBvKR: 37.9 to
15.7 MB). But to get a black-to-move rank, a reader must then read the children.

## Integrity

`payload_sha256` under `digest_scheme = sha256-tree-16MiB` is a SHA-256 hash.
Its input is the SHA-256 digests of consecutive 16 MiB payload chunks. The
payload length as u64 little-endian follows them. `content_sha256` is the
SHA-256 of the SHA-256 digests of all decompressed blocks, with sections in
header order. It identifies the content independently of codec and level.
