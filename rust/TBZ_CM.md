# `cm` section codec

This file specifies the `cm` codec of `.tbz` sections. `TBZ_FORMAT.md` gives the rest of the format.
The reference implementation is `src/cm.rs`. `src/packed.rs` has the section reader and writer.

The codec codes each stored value as a short sequence of binary decisions.
A fixed model gives each decision a probability. A binary arithmetic coder then codes the decision.
The model is in the section. Thus each block is independent of the other blocks.

## Section layout

A `cm` section has three more header fields:

- `block_words`: the stored words in each block. The writer uses 256 for `wdl` and 64 for `rank` by default.
- `model_bytes`: the length of the model blob.
- `index_bytes`: the length of the frame index.

At `offset` there is the model blob. The frame index follows it, and then the frames.
The frame index is a zstd frame. It has one u32 for each block: the length of its frame.
Frame `b` starts after the frames 0 to `b - 1`. It codes the stored words `b * block_words` to `(b + 1) * block_words - 1`.
`compressed_bytes` is the length of the model blob, the frame index and the frames.

A decoder makes the same decompressed bytes as the other codecs for each stored legal position.
For other positions, the decoder writes the previous value of the block (0 at the start of the block).
`content_sha256` uses the decompressed blocks of `block_bytes`, as for the other codecs.
Thus `content_sha256` does not change with the codec.

A `cm` rank section is only possible with `rank_sides = white`.

## Words and positions

For each stored word `cw` of a block, the decoder finds:

- The full word: `full_word(cw)`. Its skeleton squares and side to move come from the material.
- The stored mask: the stored black-king squares of the word, AND the legal mask of `statics`.

The decoder codes the positions of the stored mask from the lowest black-king square to the highest.
It does not code other positions. Their value is UNKNOWN (4) for WDL and NO_RANK for rank.

## Model blob

The model blob is a zstd frame. Its content is:

| Bytes | Content |
|---|---|
| 4 | magic `CM1\0` |
| 4 | number of models `n`, u32 |
| 4 | number of weight sets `s`, u32 |
| per model | table size `t`, u32, then a presence bitmap of `ceil(t / 8)` bytes, then one u16 for each present entry |
| 4 * s * n | weights, i32, set by set |

All integers are little-endian. Bit `i % 8` of bitmap byte `i / 8` tells if entry `i` is present.
A present entry is a probability `p` of a 1 bit, in 12 bits, from 1 to 4095.
The stretched value of a present entry is `stretch(p)`. The stretched value of an absent entry is 0.
A weight is a 16.16 fixed-point number.

## Logistic functions

`squash(x)` is the probability `4096 / (1 + e^(-x / 256))` in integers:

1. If `x > 2047`, the result is 4095. If `x < -2047`, the result is 1.
2. Let `w = x & 127` and `i = (x >> 7) + 16`, with an arithmetic shift.
3. The result is `(T[i] * (128 - w) + T[i + 1] * w + 64) >> 7`.

T has these 33 values:

```text
1 2 3 6 10 16 27 45 73 120 194 310 488 747 1101 1546 2047
2549 2994 3348 3607 3785 3901 3975 4022 4050 4068 4079 4085 4089 4092 4093 4094
```

`stretch(p)` is the smallest `x` in -2047 to 2047 with `squash(x) >= p`. If there is no such `x`, the result is 2047.

## Prediction

Each decision has one context for each model and one weight set.
The probability of a 1 bit is:

1. `dot` = the sum of `weight[set * n + i] * stretched[i][context[i]]` over the models `i`, in 64 bits.
2. `p = squash(clamp(dot >> 16, -2047, 2047))`, with an arithmetic shift.

## Arithmetic decoder

The decoder has three u32 values: `x1 = 0`, `x2 = 0xFFFFFFFF` and `x`.
At the start, `x` is the first four bytes of the frame, big-endian.
After the end of the frame, each byte is 255.

To decode one bit with probability `p`:

1. `mid = x1 + ((x2 - x1) >> 12) * p`.
2. If `x <= mid`, the bit is 1 and `x2 = mid`. Else the bit is 0 and `x1 = mid + 1`.
3. While the high bytes of `x1` and `x2` are equal: `x1 = x1 << 8`, `x2 = (x2 << 8) | 255` and `x = (x << 8) | next byte`.

The encoder writes the high byte of `x2` at each shift. At the end it writes the high byte of `x1`.

## Neighbours

`near(dw, df, dr)` is the value of the position at file `+ df` and rank `+ dr` of word `cw - dw`.
The value is UNKNOWN (or NO_RANK) in two cases.

- Word `cw - dw` is before the start of the block.
- The square is not on the board.
Words of the block are numbered from 0. A neighbour has the coded value, or UNKNOWN if the block does not code it.

The neighbours are:

| Name | dw, df, dr |
|---|---|
| left | 0, -1, 0 |
| up | 0, 0, -1 |
| ul | 0, -1, -1 |
| ur | 0, 1, -1 |
| w1, w8, w64, w512, w4096 | 1, 8, 64, 512 or 4096, then 0, 0 |
| w1l | 1, -1, 0 |
| w1u | 1, 0, -1 |

## Geometry

The skeleton has `m` pieces. Piece 0 is the white king. The black king is on square `bk`.
`attacks` and `between` are the functions of `pawnless::Geometry`.

- Occupied: the skeleton squares and `bk`.
- Piece `j` attacks square `s` if `attacks(role, square of j, occupied)` has `s`.
- Side `c` attacks piece `t` if a piece of side `c` (not `t`) attacks the square of `t`.
- For black, the black king also attacks piece `t` if the king distance from `bk` to `t` is 1.
- Piece `t` is loose if the other side attacks it and its own side does not.

For each piece `j` from 1 to `m - 1`, with bit `1 << (j - 1)`:

- `check` has the bit if `j` is white and attacks `bk`.
- `white_loose` has the bit if `j` is white and loose.
- `black_loose` has the bit if `j` is black and loose.

`kings` = clamp(king distance of piece 0 and `bk`, minus 2, 0, 2).

`geo = ((check * 16 + white_loose) * 16 + black_loose) * 3 + kings`. There are 12288 values.

`small(geo)` = `((C * 2 + W) * 2 + B) * 3 + kings`. C, W and B are 1 if `check`, `white_loose` and `black_loose` are not zero.

## WDL section

Each word has `turn` = 3 for black to move, else 0. There are 6 nodes and 30 weight sets.

### Decisions

For a position with value `v`:

1. Node 0: the bit is 1 if `v` is 0 (draw). If the bit is 1, `v` is 0.
2. Node 1: the bit is 1 if `v` is 2 (loss). If the bit is 1, `v` is 2.
3. Black to move only: node 2. The bit is 1 if `v` is 3. If the bit is 1, `v` is 3.
4. Else `v` is 1.

For node `k`, context `i` is `base[i] * 6 + k + turn`. The weight set is `(k + turn) * 5 + left`.

### Runs

The walk keeps three run lengths and the previous value in the block. At the start of a block, all are 0 and the previous value is UNKNOWN.
After each position with value `v`:

- `run_prev`: plus 1 if `v` is the previous value, else 0.
- `run_w1`: plus 1 if `v` is `w1`, else 0.
- `run_left`: plus 1 if `v` is `left`, else 0.

`bucket(r)` = min(floor(log2(r + 1)), 7).

### Contexts

Let `sq = left * 5 + up` and `wd = sq * 25 + w1 * 5 + w8`.

| Model | Size | Base |
|---|---|---|
| 0 | 1 | 0 |
| 1 | 25 | sq |
| 2 | 625 | wd |
| 3 | 15625 | wd * 25 + ul * 5 + ur |
| 4 | 15625 | wd * 25 + w64 * 5 + w512 |
| 5 | 78125 | (wd * 25 + ul * 5 + ur) * 5 + w64 |
| 6 | 3125 | (((w1 * 5 + w8) * 5 + w64) * 5 + w512) * 5 + w4096 |
| 7 | 3125 | ((sq * 5 + w1) * 5 + w1l) * 5 + w1u |
| 8 | 200 | (w1 * 8 + bucket(run_w1)) * 5 + left |
| 9 | 1000 | (sq * 8 + bucket(run_left)) * 5 + w1 |
| 10 | 200 | (previous * 8 + bucket(run_prev)) * 5 + up |
| 11 | 12288 | geo |
| 12 | 307200 | geo * 25 + sq |
| 13 | 307200 | geo * 25 + w1 * 5 + w8 |

The table size of model `i` in the blob is its size times 6.

## Rank section

The rank section covers the white-to-move words. Its block `b` uses the WDL codes of the same words.
The walk codes the rank of each stored position with WDL code 1 or 2. Let `win` be 1 for code 1, else 0.

Let `K = 8 * rank_width`. There are `N = 2 + 2 * (K + K * K)` nodes and `2 * N` weight sets.

### Prediction and decisions

The prediction `pred` is the first of `left`, `up` and `w1` that is not NO_RANK. `kind` is its place: 0, 1 or 2.
If all three are NO_RANK, `pred` is 0 and `kind` is 3. Let `d` = rank minus `pred`.

1. Node 0: the bit is 1 if `d` is 0. Then the rank is `pred`.
2. Node 1: the bit is 1 if `d` is negative. Let `off` = 2 + K * (1 + K) if the bit is 1, else 2.
3. Let `M = |d|` and `k = floor(log2(M))`. For `j` from 0: node `off + j`, the bit is 1 if `j < k`. Stop after a 0 bit or after `j = K - 2`.
4. For `i` from `k - 1` down to 0: node `off + K + k * K + i`, the bit is bit `i` of `M`.
5. `M` is 1, followed by the bits of step 4. The rank is `pred - M` or `pred + M`.

For node `k`, context `i` is `base[i] * N + k`. The weight set is `k * 2 + win`.

### Contexts

`rel(x)` is 17 if `x` is NO_RANK, else clamp(`x - pred`, -8, 8) + 8. Let `p` = min(`pred`, 256).

| Model | Size | Base |
|---|---|---|
| 0 | 8 | win * 4 + kind |
| 1 | 2056 | (win * 4 + kind) * 257 + p |
| 2 | 648 | (win * 18 + rel(up)) * 18 + rel(w1) |
| 3 | 11664 | ((win * 18 + rel(up)) * 18 + rel(ul)) * 18 + rel(ur) |
| 4 | 11664 | ((win * 18 + rel(w1)) * 18 + rel(w8)) * 18 + rel(w64) |
| 5 | 9252 | (win * 257 + p) * 18 + rel(up) |
| 6 | 11664 | ((win * 18 + rel(left)) * 18 + rel(w1)) * 18 + rel(w8) |
| 7 | 24576 | win * 12288 + geo |
| 8 | 15552 | ((win * 24 + small(geo)) * 18 + rel(left)) * 18 + rel(up) |

The table size of model `i` in the blob is its size times N.

## Encoder

The encoder can make any model. The reference encoder does these steps:

1. Count the 0 and 1 bits of each context over the full section.
2. Give a context a probability if it has 64 bits or more: `round(4096 * (n1 + 0.4) / (n + 0.8))`, clamped to 1 to 4095.
3. Train eight sets of weights on one block in eight, for two epochs, with a learn rate of 0.002. Use the mean of the eight sets.
4. Code each block in parallel.

The result does not change with the number of threads.
