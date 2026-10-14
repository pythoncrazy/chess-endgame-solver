This crate probes FENs against unchanged native Syzygy bytes through an
authenticated range filesystem. It uses the maintained `shakmaty-syzygy`
implementation pinned in Cargo.toml and Cargo.lock. It supports native WDL and
DTZ files; the CLI benchmark checks WDL after zeroing. It does not generate
tables or solve arbitrary castling/repetition-history states.

From the package root:

```sh
python download_six_references.py --directory references-six
python native_range_catalog.py --output native-objects references-six/*.rtbw
cargo test --locked --release --manifest-path rust/native/Cargo.toml --features http
python validate_six_rust.py --objects native-objects
```

The downloader verifies source sizes and SHA-256 before atomic publication.
The 37 MB comparison tables are not bundled; the small independent WDL/DTZ
fixtures needed by Cargo tests are bundled. Requires Rust 1.97 or newer;
tested with 1.99. No CUDA dependency is needed for this crate.

To probe locally with a 64 MiB retained verified-data budget:

```sh
cargo run --locked --release --manifest-path rust/native/Cargo.toml -- \
  native-objects six_extended_probe_queries.json 67108864 batch
```

`scalar` is also available. `batch` retries actual cache misses in bounded
waves, sorts/deduplicates them and coalesces adjacent physical tiles into ranges
up to 1 MiB. It adapts its window to the cache and falls back to exact scalar
probing on cache churn. This can reduce origin requests while increasing local
CPU work or fetched bytes. Compare both against the actual service workload.
Retained verified-data and index budgets are separate; they do not bound RSS,
caller outputs, concurrent transient buffers or native probe-library metadata.

The catalog binds logical table filenames to immutable content-addressed
objects. Every physical 16 KiB data tile has a SHA-256 entry; catalog-authenticated
8 KiB hash pages are fetched and cached separately. Payload/index corruption,
short responses and missing material dependencies fail instead of returning a
guessed result. The catalog is a trusted input, not a chess correctness proof.

The optional `http` feature serves the same reader over the bundled
`range_worker.mjs` resource protocol, using a user-configured deployed endpoint:

```sh
cargo run --locked --release --features http --manifest-path rust/native/Cargo.toml -- \
  native-objects six_extended_probe_queries.json 67108864 batch https://YOUR_RANGE_HOST
```

Only the local catalog is required in that mode; remote data and index `.tbobj`
objects must already be present. Nothing here uploads or deploys them. The
endpoint must return HTTP 200 and exactly the requested bytes for
`/range/<sha256>.tbobj/<offset>/<length>`. Mock HTTP transport is tested; live
R2 latency and service compute costs are unmeasured. The catalog avoids
per-probe HEAD/List requests. Warm local reads need no origin GET.
