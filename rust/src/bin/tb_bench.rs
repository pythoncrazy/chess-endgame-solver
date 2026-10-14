//! Local archive cache-pressure benchmark. Never labels local reads as R2 latency.
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    sync::atomic::Ordering,
    sync::{Arc, Mutex},
    time::Instant,
};
use tablebase_core::{
    archive::{FileStore, Manifest, RangeStore, Reader, SharedCache},
    range_cache::TileStore,
};
#[derive(Deserialize)]
struct Trace {
    batches: Vec<Vec<u64>>,
    values: Vec<Vec<i64>>,
}
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: tb_bench ARCHIVE_DIRECTORY TRACE_JSON".into());
    }
    let raw = std::fs::read(format!("{}/manifest.json", args[0])).map_err(|e| e.to_string())?;
    let trace: Trace = serde_json::from_slice(&std::fs::read(&args[1]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if trace.batches.len() != trace.values.len() {
        return Err("trace count".into());
    }
    let mut results = Vec::new();
    for aligned in [false, true] {
        let manifest = Manifest::from_json(&raw)?;
        let origin = Arc::new(FileStore::new(&args[0]));
        let transport: Arc<dyn RangeStore> = if aligned {
            let sizes: BTreeMap<_, _> = manifest
                .objects
                .iter()
                .map(|o| (o.key.clone(), o.bytes))
                .collect();
            Arc::new(TileStore::new(
                origin.clone(),
                sizes,
                16384,
                Arc::new(Mutex::new(SharedCache::new(1 << 20))),
            )?)
        } else {
            origin.clone()
        };
        let cache = Arc::new(Mutex::new(SharedCache::new(128 << 10)));
        let reader = Reader::new(manifest, transport, cache.clone());
        let mut timings = Vec::new();
        for (ids, expected) in trace.batches.iter().zip(&trace.values) {
            let start = Instant::now();
            let values = reader.probe_many(ids)?;
            timings.push(start.elapsed().as_nanos());
            if &values != expected {
                return Err("independent oracle mismatch".into());
            }
        }
        timings.sort_unstable();
        results.push(serde_json::json!({"aligned_compressed_tiles":aligned,"batches":trace.batches.len(),"queries":trace.batches.iter().map(Vec::len).sum::<usize>(),"origin_reads":origin.reads.load(Ordering::Relaxed),"origin_bytes":origin.bytes.load(Ordering::Relaxed),"median_batch_ns":timings[timings.len()/2],"retained_core_cache_bytes":cache.lock().unwrap().used()}));
    }
    println!(
        "{}",
        serde_json::json!({"results":results,"environment":"local CPU and filesystem; no network or GPU; warm operating-system page cache possible"})
    );
    Ok(())
}
