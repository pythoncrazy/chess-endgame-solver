//! CPU six-piece FEN benchmark over authenticated, cached native range reads.
use serde::Deserialize;
use shakmaty::{CastlingMode, Chess, fen::Fen};
use shakmaty_syzygy::Tablebase;
use std::{
    sync::atomic::Ordering,
    sync::{Arc, Mutex},
    time::Instant,
};
use tablebase_core::{
    archive::{FileStore, RangeStore, SharedCache},
    range_cache::TileStore,
};
use tablebase_native::{Catalog, FilesystemHandle, NativeFilesystem, transport::Metered};
#[derive(Deserialize)]
struct Queries {
    fens: Vec<String>,
    values: Vec<i32>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 2 || args.len() > 5 {
        return Err("usage: tablebase-native OBJECT_DIRECTORY QUERY_JSON [CACHE_BYTES] [scalar|batch] [RANGE_BASE_URL]".into());
    }
    let catalog = Catalog::parse(&std::fs::read(format!("{}/native-catalog.json", args[0]))?)?;
    let selected: Arc<dyn RangeStore> = if let Some(url) = args.get(4) {
        #[cfg(feature = "http")]
        {
            Arc::new(tablebase_native::transport::HttpRanges::new(url)?)
        }
        #[cfg(not(feature = "http"))]
        {
            let _ = url;
            return Err("enable the http Cargo feature for a remote endpoint".into());
        }
    } else {
        Arc::new(FileStore::new(&args[0]))
    };
    let origin = Arc::new(Metered::new(selected));
    // Single-flight with no second cached copy of the authenticated data.
    let transport = Arc::new(TileStore::new(
        origin.clone(),
        catalog.object_sizes(),
        16384,
        Arc::new(Mutex::new(SharedCache::new(0))),
    )?);
    let cache_budget = args
        .get(2)
        .map(|x| x.parse::<usize>())
        .transpose()?
        .unwrap_or(64 << 20);
    let mode = args.get(3).map(String::as_str).unwrap_or("scalar");
    if !["scalar", "batch"].contains(&mode) {
        return Err("mode must be scalar or batch".into());
    }
    let filesystem = Arc::new(NativeFilesystem::new(
        catalog,
        transport,
        cache_budget,
        4 << 20,
    ));
    let mut tables: Tablebase<Chess> =
        Tablebase::with_filesystem(Arc::new(FilesystemHandle(filesystem.clone())));
    for name in filesystem.files.keys() {
        tables.add_file(name)?;
    }
    let query: Queries = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    if query.fens.len() != query.values.len() {
        return Err("query count mismatch".into());
    }
    let positions: Vec<Chess> = query
        .fens
        .iter()
        .map(|fen| {
            fen.parse::<Fen>()
                .map_err(|e| e.to_string())
                .and_then(|fen| {
                    fen.into_position(CastlingMode::Standard)
                        .map_err(|e| e.to_string())
                })
        })
        .collect::<Result<_, _>>()?;
    let mut results = Vec::new();
    for phase in ["cold", "warm"] {
        let before = origin.reads.load(Ordering::Relaxed);
        let bytes = origin.bytes.load(Ordering::Relaxed);
        let start = Instant::now();
        let mut durations = Vec::new();
        let (waves, fallbacks) = if mode == "batch" {
            let (values, waves, fallbacks) = filesystem.probe_many(&tables, &positions, 128)?;
            if values != query.values {
                return Err("native batch oracle mismatch".into());
            }
            (waves, fallbacks)
        } else {
            for (position, &expected) in positions.iter().zip(&query.values) {
                let t = Instant::now();
                let actual = tables.probe_wdl_after_zeroing(position)? as i32;
                durations.push(t.elapsed().as_nanos());
                if actual != expected {
                    return Err(format!("published WDL mismatch: {actual} vs {expected}").into());
                }
            }
            (0, 0)
        };
        let wall = start.elapsed().as_secs_f64();
        durations.sort_unstable();
        results.push(serde_json::json!({"phase":phase,"positions":positions.len(),"seconds":wall,"positions_per_second":positions.len() as f64/wall,"median_position_ns":durations.get(durations.len()/2),"p95_position_ns":durations.get(durations.len()*95/100),"dependency_waves":waves,"scalar_fallbacks":fallbacks,"origin_reads":origin.reads.load(Ordering::Relaxed)-before,"origin_bytes":origin.bytes.load(Ordering::Relaxed)-bytes}));
    }
    let (data, index) = filesystem.retained_bytes()?;
    println!(
        "{}",
        serde_json::json!({"results":results,"mode":mode,"cache_budget_bytes":cache_budget,"retained_data_cache_bytes":data,"retained_index_cache_bytes":index,"logical_read_hints":filesystem.read_hints.iter().map(|x|x.load(Ordering::Relaxed)).collect::<Vec<_>>(),"exact_published_wdl_at_reset":true,"native_compression_preserved":true,"h100_generation_benchmark":false,"environment":"local CPU and filesystem; positions pre-parsed; not cloud transport latency"})
    );
    Ok(())
}
