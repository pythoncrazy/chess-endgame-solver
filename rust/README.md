Rust is the production implementation target. The Python implementations remain
independent correctness oracles and the current archive publisher/FEN adapter.

The implemented Rust portion is an exact page encoder, a native position-ID
prober, authenticated range-readable indexes, global cache budgets, stable
physical range tiles, concurrent single-flight, and finite-graph WDL oracles.
This is a tested foundation, not a complete eight-piece generator or Rust FEN
prober. The Rust CLI accepts archive IDs; its Python counterpart ranks FENs and
resolves captures/promotions against the published Syzygy semantics.

From this directory:

```sh
cargo test --locked --release
cargo clippy --locked --all-targets -- -D warnings
cargo run --locked --release --bin tb_probe -- ../bridge_final_example 0 1 99
cargo run --locked --release --bin tb_bench -- ../pawn_examples/KPvKP fixtures/cache_trace.json
printf '[0,-2,1,9223372036854775807]' | cargo run --locked --release --bin tb_encode -- example.tbc
```

`Cargo.lock` pins CPU dependencies. Tested with Rust 1.99.0 on Linux; the declared
1.89 minimum has not separately been tested. Own CPU Rust code forbids unsafe.
The compression crates use native codec libraries internally.

`../validate_rust.py` runs formatting, Clippy, release tests and benchmarks, then
feeds Rust-encoded pages to the independent Python decoder. The bundled fixtures
are sufficient for Cargo tests; Python is not required just to run those tests.
`../export_rust_fixtures.py` documents how those independent fixtures were made.

Cache budgets charge retained bodies, keys and a conservative entry allowance.
They are not process RSS limits: caller output, grouping, decompressor buffers,
allocator overhead and values borrowed by concurrent queries are additional.
Limit admitted request/batch concurrency in a service. Reader windows default
to 64 MiB of worst-case decoded cells plus one page. Individual origin requests
are capped at 1 MiB; frame count, encoded bytes and codec windows are bounded.
Catalog metadata is limited to 1,048,576 pages per object. A future complete
collection needs a hierarchical material/object catalog, rather than a single
in-memory vector of every object.

The supplied trusted catalog authenticates directories, index chunks and v3
encoded frames. Selective reads rely on a writer that verifies full pages before
publication. Authentication proves correspondence to that catalog; it does not
prove the publisher solved chess correctly. Legacy v2 frames undergo full decoded
checksums. Predictive modes 4/5 require an explicit predictor in the page API;
archive Reader rejects their missing predictor instead of guessing a result.

GPU development uses the **current NVIDIA/cuda-rust project**, whose Tile crate
is named `cutile`. `gpu/Cargo.toml` pins revision
`bc46b1ffb751885073066b087594045f07f0f81b`. The independent GPU workspace avoids
building CUDA dependencies for CPU storage tests. Its source uses the inspected
API and has been formatted, but GPU host/JIT compilation is **unverified**:
the attempted build stopped in cuda-bindings because this environment has no
CUDA toolkit. The GPU lockfile records the resolved dependency graph.

On the H100 machine, install a compatible driver and CUDA toolkit with Hopper
Tile IR support (13.3 minimum Tile IR in this pinned upstream), then:

```sh
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.3 cargo run --locked --release --manifest-path gpu/Cargo.toml
```

This benchmark validates one million synthetic rows against the CPU predicate
before timing. Inputs are already-built 64-child masks. It excludes move
generation, rank computation, dependency traffic, compression and distances.
Timing includes host launch/sync overhead. Nominal traffic is 32 bytes/row;
cache reuse means it is not an HBM bandwidth measurement. More than 64 children
require blockwise OR of the win predicate and AND of the all-winning predicate.
Unknown/cyclic positions must remain unresolved until the global exact fixpoint.

Use Tile for dense, regular operators and benchmark SIMT for irregular move
generation/frontier traffic. Both NVIDIA models share a runtime; using Rust or
cuTile alone does not establish which kernel is faster. Keep GPU queues/counters
and archive IDs integer. FP16 inputs/FP32 tensor reductions are permissible only
for bounded, exactly representable arithmetic, with a scalar/bitset oracle and
measured useful occupancy. No one-hour full-eight result is claimed.
