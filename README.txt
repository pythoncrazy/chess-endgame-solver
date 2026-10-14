H100 HANDOFF UPDATE — 2026-10-14 UTC

Read h100_agent_handoff.txt for the complete machine procedure.
This package contains the subsequent native reader, dense6 index and FP8
operator comparison sources. The earlier version 3 ZIP did not contain them.
This package also contains the CPU test records. GPU host/JIT compilation and
H100 execution are not verified.

EXACT TABLEBASE RESEARCH LAB — 2026-10-13 UTC

This package contains a design report, validated CPU reference rules, capacity
models and a GPU operator microbenchmark. It is NOT a complete chess generator.
It does NOT contain a generated nine-piece tablebase. We did NOT benchmark it
on an H100 in this environment.

Read report.html first. Sections 19–21 cover the latest edge-case and Rust work.

RUST IMPLEMENTATION (production target)
  - rust/: exact page encoder and selective authenticated lookup by native ID.
    It also has v2/v3 indexes, global cache budgets, coalesced reads and
    aligned single-flight ranges.
  - rust/gpu/: cuTile microbenchmark source for bitmask operators, pinned to
    a NVIDIA/cuda-rust revision.
  - rust_validation.json: release/Clippy/format/independent-oracle results.
  - rust_cache_benchmark.json: local 2,048-query cache-pressure measurements.
  Python is still the independent oracle and the current archive/FEN publisher.
  The Rust code does NOT yet calculate FEN ranks or contain a complete GPU
  generator.

Run these commands from this package directory:
    cargo test --locked --release --manifest-path rust/Cargo.toml
    cargo clippy --locked --manifest-path rust/Cargo.toml --all-targets -- -D warnings
    cargo run --locked --release --manifest-path rust/Cargo.toml --bin tb_probe \
      -- bridge_final_example 0 1 99
    python validate_rust.py
See rust/README.md for GPU setup, actual build status and memory contracts.
We tried the GPU build here, but absent CUDA headers/toolkit stopped it.
We did not do a host/JIT compilation of the source. We did not execute the
source on an H100.

LATEST ADDITIONS
  - exact_quotient.py and validate_quotient.py: experiments with exact position
    groups and a complete canonical KQvK solve, with optional independent
    Syzygy comparison.
  - r2_blocks.py: two-bit WDL + Zstandard frames, range reads and grouped queries.
  - full8_budget.py: whole-collection one-hour budgets and R2 cost scenarios.
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
The --chess-path argument adds that directory at the start of the import path.
If no local chess.py exists, Python then imports the installed python-chess
package normally.

Capacity model:
    python rooflines.py

GPU experiment on your CUDA GPU (install a compatible PyTorch first):
    python -m pip install numpy triton
    python benchmark_h100.py --groups 8192 --output h100_benchmark.json

The GPU experiment validates resident synthetic move-relation operators before
it measures their time. It excludes relation construction, chess index
calculation, scheduler work, pack operations, I/O and distances. The queries=16
case must have 16 useful vectors that share one legal relation. Do not use it as
a measure of general chess throughput.

reference.py validates ordinary WDL on finite graphs. It does not validate the
full FIDE game with repetition history. It is not a production DTZ50
implementation. Only optional chess-fiber tests use python-chess. This package
does not contain python-chess.

LATEST STORAGE AND LOOKUP IMPLEMENTATION
  - adaptive_storage.py: exact codec selection for each signed-64-bit block,
    with constant, palette, residual and predictor+exception codecs. It does
    not round values.
  - lookup_storage.py: paged authenticated indexes, bounded caches, coalesced
    byte ranges, compact decoded WDL and optional bounded parallel range reads.
  - singleflight_store.py: deduplication of concurrent immutable range fetches.
  - native_wdl_bridge.py: complete KQvKR WDL archive with batched FEN probes.
  - range_worker.mjs: read-only proxy with an R2 range cache. It is NOT DEPLOYED.
  - lookup_costs.py: R2 + Workers cost model with explicit traffic assumptions.
  - bridge_final_example/: current v3 complete KQvKR WDL and bound chess metadata.
  - bridge_fast_example/: legacy v2 compatibility fixture.
  - pawn_examples/: complete KPvK/KPvKP WDL native streams and rank metadata.
  - packed_lookup.py: authenticated selective extraction without full unpack.
  - chess_state.py: lossless board/game context. It is not a full FIDE-history
    solver.
  - archive_session.py: reusable FEN adapter with shared per-session cache budgets.

QUICK START (a GPU or cloud account is not necessary)
    python -m pip install -r requirements.txt
    python download_reference_tables.py --directory references
    python native_wdl_bridge.py --chess-path . --table-path references \
      --archive-path bridge_final_example \
      --fen '8/8/8/8/8/7k/4r3/KQ6 w - - 0 1'

The example returns the published Syzygy WDL class 2. For other materials and
DTZ, the downloaded local dependencies are still necessary.

The archive replaces the original KQvKR WDL payload. The archive gives its chess
rank metadata explicitly. This example uses private python-chess interfaces and
is pinned to chess 1.11.2.

Add --metric dtz to keep the published DTZ probe semantics. The original local
DTZ data is still necessary. The values do not solve arbitrary repetition
history.

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
The benchmark makes a temporary cache of reference symbols outside this package.
A complete KQvKR extraction can take several minutes on CPU. Subsequent runs use
that cache again. In storage_benchmark.json, the labels show that the other
four-piece materials are samples, not full table measurements.

FUTURE CLOUD CONNECTION
Bind your R2 bucket as TB_BUCKET in range_worker.mjs. Deploy range_worker.mjs
through your own Cloudflare project. Nothing in this package uploads or deploys
data.

The Worker serves only immutable .tbobj ranges. It does not decode chess data or
give a public FEN API. After the archive objects are present and you deploy that
endpoint, the CLI accepts --range-base-url https://YOUR_RANGE_HOST. The local
manifest/chess metadata and other-material dependencies are still necessary.
S3RangeStore also supports an already-configured S3 client.

Cache API data is local to a Cloudflare data center. The Worker wraps a range as
an HTTP-200 resource because cache.put rejects HTTP-206 responses. Do not assume
that this proxy is a tiered cache. Do not assume that a client request itself
becomes free. Deployment measurements are still necessary for actual cloud
latency, traffic distribution, memory limits, cold starts and fleet-wide cache
behavior.
