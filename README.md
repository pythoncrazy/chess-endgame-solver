# Chess endgame solver

This repository makes exact tables for pawnless chess endgames.
A table gives the result of each position with best play: win, draw or loss.
It also gives a proof rank. This is the number of plies to checkmate, or to a
capture into a table with fewer pieces.

The solver runs on one NVIDIA H100 GPU. A CPU solver gives the same tables
for comparison. A separate program checks each table against the published
Syzygy tables. The tables are stored in a compressed format, and a small
web viewer reads them.

## Results

All tables are pawnless. They do not include castling rights.

| Set | Tables | Format | Size on disk |
|---|---|---|---|
| All materials with five pieces or fewer | 95 | `.tbz` v4, zstd | 644,059,376 bytes |
| The same tables | 95 | `.tbz` v4, lzma | 581,458,543 bytes |
| KQBNvKR and its capture tree (six pieces) | 16 | `.tbp` v1 | 72,423,859,154 bytes |

The `verify_pawnless` program makes these checks on each position:

- The legality of the position agrees with shakmaty.
- The win, draw or loss agrees with the Syzygy WDL value.
- The proof rank is exact. The program calculates it again from the ranks of the next positions.
- With `--dtz`, the proof rank is equal to the Syzygy DTZ value, or one more. A Syzygy DTZ
  can be one ply short. This is a second check of the ranks. It does not use the solver
  data of the next positions.

Thus a table cannot pass with a wrong result or with a wrong rank.

## Contents

| Path | Contents |
|---|---|
| `rust/src/` | The core library: the CPU solver (`pawnless.rs`), the `.tbz` format (`packed.rs`) and the context-model codec (`cm.rs`). |
| `rust/src/bin/` | `tb_solve` (CPU solver), `tb_pack` (writes `.tbz` files), `tb_compare` (compares two tables), `tb_entropy` (compression study). |
| `rust/gpu/` | `gpu_solve`, the GPU solver. It uses cuTile Rust. |
| `rust/verify/` | `verify_pawnless`, the independent check with shakmaty and Syzygy. |
| `tools/visualizer/` | The web viewer. |
| `tools/` | Python tools: Syzygy download, online comparison, table digests, the solver checks, the prose check and the commit-message check. `logrun.sh` records the output of long jobs. |
| `docs/` | The `.tbz` format, the `cm` codec and the compression experiments. |

## Requirements

- Rust. The file `rust-toolchain.toml` sets the version.
- [uv](https://docs.astral.sh/uv/) for the Python tools.
- For the GPU solver only: an NVIDIA H100 GPU and the CUDA 13.3 toolkit.

## Procedures

Set the data directory first. The tools write tables, references and logs there.

```sh
export TB_ROOT=/path/to/data
```

### Get the reference tables

Download the Syzygy WDL tables for all pawnless materials with five pieces or fewer.
Add `--dtz` to download the DTZ tables also (290 MB):

```sh
uv sync
uv run tools/fetch_syzygy.py "$TB_ROOT/references-345" 5 --dtz
```

### Solve tables

To solve one material and its capture tree on the CPU:

```sh
cargo run --release --manifest-path rust/Cargo.toml --bin tb_solve -- KQvKR "$TB_ROOT/results/cpu"
```

To solve all materials with five pieces or fewer on the GPU:

```sh
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3
cargo run --release --manifest-path rust/gpu/Cargo.toml --bin gpu_solve -- --all 5 "$TB_ROOT/results/all5"
```

The GPU solver writes `.tbz` files with `--all`. For one material, it writes `.gpu.tbp` files.
With `--all`, you can give the zstd level (default 15) and the number of pack
threads (default 20). Level 19 makes files approximately 2% smaller,
but the compression takes approximately three times longer.

### Pack tables

To write `.tbz` files from `.tbp` files:

```sh
cargo run --release --manifest-path rust/Cargo.toml --bin tb_pack -- "$TB_ROOT/results/packed" "$TB_ROOT"/results/cpu/*.tbp
```

Use `--wdl-codec` and `--rank-codec` to select `zstd`, `lzma` or `cm`.

### Verify tables

Give the program each table of the capture tree:

```sh
cargo run --release --manifest-path rust/verify/Cargo.toml --bin verify_pawnless -- \
  "$TB_ROOT/references-345" 24 "$TB_ROOT"/results/all5/*.tbz
```

The second argument is the number of threads. Add `--stored` to check one position
of each symmetry class and the symmetry of the table. This is faster than a check
of all positions. Add `--dtz` to compare the proof ranks with the Syzygy DTZ tables.

### Open the viewer

```sh
tools/visualizer/run.sh 8642
```

Then open `http://127.0.0.1:8642/` in a browser.

## Development

Install the hooks once:

```sh
uv sync
uv run prek install
```

The hooks and the CI make these checks:

- `cargo fmt`, `cargo clippy -D warnings` and `cargo test` for the three Rust crates.
- `ruff check` and `ruff format` for the Python tools.
- `tools/ste_check.py` for the prose. All prose obeys ASD-STE100 Simplified Technical English.
- `tools/strip_trailers.py` for the commit messages. They must not have co-author or tool lines.

After a change to the GPU solver, do the solver checks. Each check solves a target
and compares the digests of the tables with `tools/expected_digests.json`:

```sh
cargo build --release --manifest-path rust/gpu/Cargo.toml --bin gpu_solve
uv run tools/check_solver.py KQvKR tmp/check-KQvKR
uv run tools/check_solver.py KQBvKR tmp/check-KQBvKR
uv run tools/check_solver.py all5 tmp/check-all5 28
```

Each output directory must be empty. The last argument is the number of pack threads.
On an H100 NVL, the `all5` check takes approximately 50 s.

Use short commit messages in the Conventional Commits form, for example
`fix(packed): reject a bad header`. Make each change in a pull request. Merge with a rebase.

Put temporary scripts and outputs in `tmp/`. Git ignores that directory.
