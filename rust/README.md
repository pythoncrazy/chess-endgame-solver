Rust is the target for the production implementation. The Python implementations
stay as independent correctness oracles. Python is now also the archive
publisher and the FEN adapter.

The Rust part that exists now has an exact page encoder and a native
position-ID prober. It also has authenticated range-readable indexes, global
cache budgets and stable physical range tiles. It also has concurrent
single-flight and finite-graph WDL oracles. This is a tested foundation, not a
complete eight-piece generator or Rust FEN prober. The Rust CLI accepts archive
IDs. Its Python counterpart ranks FENs. It also resolves captures and promotions
against the published Syzygy semantics.

From this directory:

```sh
cargo test --locked --release
cargo clippy --locked --all-targets -- -D warnings
cargo run --locked --release --bin tb_probe -- ../bridge_final_example 0 1 99
cargo run --locked --release --bin tb_bench -- ../pawn_examples/KPvKP fixtures/cache_trace.json
printf '[0,-2,1,9223372036854775807]' | cargo run --locked --release --bin tb_encode -- example.tbc
```

`Cargo.lock` pins the CPU dependencies. The tests used Rust 1.99.0 on Linux. No
separate test used the declared minimum, 1.89. The CPU Rust code of this project
forbids unsafe code. The compression crates use native codec libraries
internally.

`../validate_rust.py` runs the format check, Clippy, the release tests and the
benchmarks. Then it sends Rust-encoded pages to the independent Python decoder.
The bundled fixtures are enough for the Cargo tests. Python is not necessary to
run those tests. `../export_rust_fixtures.py` is the script that made those
independent fixtures.

Cache budgets count the kept bodies, the keys and a conservative fixed size for
each entry. They do not limit the process RSS. The budget excludes
caller output, query groups, decompressor buffers, allocator overhead
and values that concurrent queries borrow. In a service, limit the number of
concurrent requests and batches. The default reader window is 64 MiB of
worst-case decoded cells plus one page. Each origin request has a maximum of 1
MiB. The frame count, the encoded bytes and the codec windows also have limits.
Catalog metadata has a limit of 1,048,576 pages for each object. A complete
collection in the future must have a hierarchical catalog of materials and
objects. A single in-memory vector of all objects is not enough.

The trusted catalog that the caller gives authenticates directories, index
chunks and v3 encoded frames. Selective reads rely on a writer that verifies
full pages before publication. Authentication proves that the data agrees with
that catalog. It does not prove that the publisher solved chess correctly.
Legacy v2 frames get a full checksum of the decoded data. Predictive modes 4
and 5 must have an explicit predictor in the page API. If the predictor is
absent, the archive Reader rejects the frame and does not guess a result.

GPU development uses the **current NVIDIA/cuda-rust project**. Its Tile crate
has the name `cutile`. `gpu/Cargo.toml` pins revision
`bc46b1ffb751885073066b087594045f07f0f81b`. The GPU workspace is independent,
so the CPU storage tests do not build CUDA dependencies. The GPU workspace
builds and runs on an H100 NVL with CUDA 13.3. CI does not build it, because CI
has no CUDA toolkit. The GPU lockfile records the resolved dependency graph.

On the H100 machine, install a compatible driver and a CUDA toolkit with Hopper
Tile IR support. This pinned upstream must have Tile IR 13.3 or later. Then run:

```sh
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 cargo run --locked --release --manifest-path gpu/Cargo.toml
```

This benchmark validates one million synthetic rows against the CPU predicate
before it measures time. Inputs are already-built 64-child masks. It excludes
move generation, rank computation, dependency traffic, compression and
distances. The measured time includes the host launch and sync overhead. Nominal
traffic is 32 bytes for each row. Because of cache reuse, the result is not a
measurement of HBM bandwidth. Over 64 children, use a blockwise OR of the win
predicate and a blockwise AND of the all-win predicate. Unknown and
cyclic positions must stay unresolved until the global exact fixpoint.

Use Tile for dense, regular operators and benchmark SIMT for irregular move
generation/frontier traffic. Both NVIDIA models share a runtime. The choice of
Rust or cuTile alone does not show which kernel is faster. Keep GPU
queues/counters and archive IDs integer. Use FP16 inputs or FP32 tensor
reductions only for bounded arithmetic that they represent exactly. Then also
use a scalar or bitset oracle, and measure the useful occupancy. This project
does not claim a one-hour result for the full eight-piece set.
