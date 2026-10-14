H100 HANDOFF UPDATE — 2026-10-14 UTC

Read h100_agent_handoff.txt for the complete machine procedure.
This package includes the later native reader, dense6 index and FP8 operator
comparison sources missing from the earlier version 3 ZIP. CPU test records
are included. GPU host/JIT compilation and H100 execution remain unverified.

EXACT TABLEBASE RESEARCH LAB — 2026-10-13 UTC

This package contains a design report, validated CPU reference rules, capacity
models and a GPU operator microbenchmark. It is NOT a complete chess generator,
does NOT contain a generated nine-piece tablebase, and has NOT been benchmarked
on an H100 in this environment.

Read report.html first. Sections 19–21 cover the latest edge-case and Rust work.

RUST IMPLEMENTATION (production target)
  rust/: exact page encoder, selective authenticated native-ID lookup, v2/v3
    indexes, global cache budgets, coalescing and aligned single-flight ranges.
  rust/gpu/: pinned NVIDIA/cuda-rust cuTile bitmask microbenchmark source.
  rust_validation.json: release/Clippy/format/independent-oracle results.
  rust_cache_benchmark.json: local 2,048-query cache-pressure measurements.
  Python remains the independent oracle and current archive/FEN publisher.
  Rust FEN ranking and a complete GPU generator are NOT implemented yet.

From this package directory:
  cargo test --locked --release --manifest-path rust/Cargo.toml
  cargo clippy --locked --manifest-path rust/Cargo.toml --all-targets -- -D warnings
  cargo run --locked --release --manifest-path rust/Cargo.toml --bin tb_probe \
    -- bridge_final_example 0 1 99
  python validate_rust.py
See rust/README.md for GPU setup, actual build status and memory contracts.
The GPU build was attempted and blocked by absent CUDA headers/toolkit here;
the source has not been host/JIT compiled or executed on an H100.

LATEST ADDITIONS:
  exact_quotient.py and validate_quotient.py: exact grouping experiments and
  complete canonical KQvK solving, with optional independent Syzygy comparison.
  r2_blocks.py: two-bit WDL + Zstandard frames, range reads and grouped queries.
  full8_budget.py: whole-collection one-hour budgets and R2 cost scenarios.
  Read the new sections of report.html for measured results and limitations.

Storage validation:
  python -m pip install zstandard
  python r2_blocks.py

Quotient validation:
  python validate_quotient.py --chess-path .
Add --table-path PATH if that folder contains KQvK.rtbw for comparison.

CPU checks:
  python -m pip install numpy python-chess
  OPENBLAS_NUM_THREADS=1 python validate.py --chess-path . --output validation.json
The --chess-path argument prepends that directory; an installed python-chess
package is then imported normally if no local chess.py exists.

Capacity model:
  python rooflines.py

GPU experiment (on your CUDA GPU, after installing a compatible PyTorch):
  python -m pip install numpy triton
  python benchmark_h100.py --groups 8192 --output h100_benchmark.json

The GPU experiment validates resident synthetic move-relation operators before
timing them. It excludes relation construction, chess indexing, scheduling,
packing, I/O and distances. The queries=16 case requires 16 useful vectors
sharing one legal relation; it must not be treated as general chess throughput.

reference.py validates ordinary WDL on finite graphs, not the entire FIDE game
with repetition history or a production DTZ50 implementation. Python-chess is
used only by optional chess-fiber tests. It is not bundled in this package.

LATEST STORAGE AND LOOKUP IMPLEMENTATION
  adaptive_storage.py: exact signed-64-bit per-block codec selection, including
    constant, palette, residual and predictor+exception encodings. No rounding.
  lookup_storage.py: paged authenticated indexes, bounded caches, byte-range
    coalescing, compact decoded WDL and optional bounded parallel range reads.
  singleflight_store.py: concurrent immutable range fetch deduplication.
  native_wdl_bridge.py: complete KQvKR WDL archive with batched FEN probing.
  range_worker.mjs: read-only R2 range cache proxy; NOT DEPLOYED.
  lookup_costs.py: R2 + Workers cost model with explicit traffic assumptions.
  bridge_final_example/: current v3 complete KQvKR WDL and bound chess metadata.
  bridge_fast_example/: legacy v2 compatibility fixture.
  pawn_examples/: complete KPvK/KPvKP WDL native streams and ranking metadata.
  packed_lookup.py: authenticated selective extraction without full unpack.
  chess_state.py: lossless board/game context; not a full FIDE-history solver.
  archive_session.py: reusable FEN adapter with shared per-session cache budgets.

QUICK START (no GPU or cloud account required)
  python -m pip install -r requirements.txt
  python download_reference_tables.py --directory references
  python native_wdl_bridge.py --chess-path . --table-path references \
    --archive-path bridge_final_example \
    --fen '8/8/8/8/8/7k/4r3/KQ6 w - - 0 1'

The example returns the published Syzygy WDL class 2. Other materials and DTZ
still need the downloaded local dependencies. The original KQvKR WDL payload
is replaced by the archive; its chess ranking metadata is supplied explicitly.
This uses private python-chess interfaces and is pinned to chess 1.11.2.
Add --metric dtz to preserve published DTZ probing semantics; original local
DTZ data remains required. Values do not solve arbitrary repetition history.

CURRENT CHECKS
  python validate_adaptive_storage.py
  python validate_lookup_storage.py
  python singleflight_store.py
  node validate_range_worker.mjs
  python lookup_costs.py
  python validate_storage_edges.py
  python validate_packed_lookup.py
  python validate_chess_edges.py --chess-path . --table-path references \
    --pawn-archives pawn_examples
  python validate_archive_session.py --chess-path . --table-path references \
    --pawn-archives pawn_examples
  python download_reference_tables.py --directory references --extended
  python validate_cursed_classes.py --chess-path . --table-path references

REPRODUCE CHESS/COMPRESSION MEASUREMENTS
  python benchmark_storage.py --chess-path . --table-path references
  python sweep_storage.py
  python native_wdl_bridge.py --chess-path . --table-path references \
    --archive-path YOUR_NEW_OUTPUT_DIR --page-positions 262144
The benchmark creates a temporary reference-symbol cache outside this package.
Complete KQvKR extraction can take several minutes on CPU; subsequent runs reuse
that cache. Other four-piece materials in storage_benchmark.json are labelled
samples, not full table measurements.

FUTURE CLOUD CONNECTION
Bind an existing R2 bucket as TB_BUCKET in range_worker.mjs and deploy it through
your own Cloudflare project. The Worker serves only immutable .tbobj ranges;
it does not implement chess decoding or a public FEN API. After the archive
objects are present and that endpoint is deployed, the CLI accepts
--range-base-url https://YOUR_RANGE_HOST. The local manifest/chess metadata and
other-material dependencies are still required. S3RangeStore also supports an
already-configured S3 client. Nothing in this package uploads or deploys data.

Cache API data is local to a Cloudflare data center. A range is wrapped as an
HTTP-200 resource because cache.put rejects HTTP-206 responses. Do not assume
this proxy is tiered caching, or that a client request itself becomes free.
Actual cloud latency, traffic distribution, memory limits, cold starts and
fleet-wide cache behavior still require deployment measurements.
