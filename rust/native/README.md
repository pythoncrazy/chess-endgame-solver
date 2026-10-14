This crate probes FENs against unchanged native Syzygy bytes through an
authenticated range filesystem. It uses the maintained `shakmaty-syzygy`
implementation that Cargo.toml and Cargo.lock pin. It supports native WDL and
DTZ files. The CLI benchmark checks WDL after zeroing. It does not generate
tables or solve arbitrary castling/repetition-history states.

From the package root:

```sh
python download_six_references.py --directory references-six
python native_range_catalog.py --output native-objects references-six/*.rtbw
cargo test --locked --release --manifest-path rust/native/Cargo.toml --features http
python validate_six_rust.py --objects native-objects
```

The downloader verifies source sizes and SHA-256 before atomic publication.
The 37 MB comparison tables are not in the bundle. The bundle has the small
independent WDL/DTZ fixtures for the Cargo tests. This crate must have Rust 1.97
or newer. The tests used 1.99. This crate has no CUDA dependency.

To probe locally with a 64 MiB budget for kept verified data:

```sh
cargo run --locked --release --manifest-path rust/native/Cargo.toml -- \
  native-objects six_extended_probe_queries.json 67108864 batch
```

`scalar` is also available. `batch` retries actual cache misses in bounded
waves. It sorts them, removes duplicates and merges adjacent physical tiles into
ranges of up to 1 MiB. It adapts its window to the cache. On cache churn, it
falls back to exact scalar probes. This can decrease origin requests, but it can
increase local CPU work or fetched bytes. Compare both against the actual
service workload. The budgets for kept verified data and for the index are
separate. They do not limit RSS, caller outputs, concurrent transient buffers or
native probe-library metadata.

The catalog binds logical table filenames to immutable content-addressed
objects. Every physical 16 KiB data tile has a SHA-256 entry. The reader fetches
and caches the catalog-authenticated 8 KiB hash pages separately. Payload or
index corruption, short responses and absent material dependencies cause a
failure. The reader does not return a guessed result. The catalog is a trusted
input, not a proof of chess correctness.

The optional `http` feature serves the same reader over the bundled
`range_worker.mjs` resource protocol. It uses a deployed endpoint that the user
configures:

```sh
cargo run --locked --release --features http --manifest-path rust/native/Cargo.toml -- \
  native-objects six_extended_probe_queries.json 67108864 batch https://YOUR_RANGE_HOST
```

In that mode, only the local catalog is necessary. The remote data and index
`.tbobj` objects must already be present. Nothing here uploads or deploys them.
The endpoint must return HTTP 200 and exactly the requested bytes for
`/range/<sha256>.tbobj/<offset>/<length>`. The tests cover a mock HTTP
transport. There are no measurements of live R2 latency or service compute
costs. The catalog avoids per-probe HEAD/List requests. Warm local reads send no
GET to the origin.
