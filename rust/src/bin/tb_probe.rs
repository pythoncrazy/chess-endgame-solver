//! Probe native archive position IDs. This CLI does not rank FENs.
use std::{
    sync::atomic::Ordering,
    sync::{Arc, Mutex},
    time::Instant,
};
use tablebase_core::archive::{FileStore, Manifest, Reader, SharedCache};
fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let directory = args
        .next()
        .ok_or("usage: tb_probe ARCHIVE_DIRECTORY POSITION_ID ...")?;
    let positions: Vec<u64> = args
        .map(|x| x.parse::<u64>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    let raw = std::fs::read(format!("{directory}/manifest.json")).map_err(|e| e.to_string())?;
    let store = Arc::new(FileStore::new(&directory));
    let cache = Arc::new(Mutex::new(SharedCache::new(64 << 20)));
    let reader = Reader::new(Manifest::from_json(&raw)?, store.clone(), cache.clone());
    let start = Instant::now();
    let values = reader.probe_many(&positions)?;
    let elapsed = start.elapsed().as_micros();
    let cold_reads = store.reads.load(Ordering::Relaxed);
    let start = Instant::now();
    if reader.probe_many(&positions)? != values {
        return Err("warm mismatch".into());
    }
    println!(
        "{}",
        serde_json::json!({"values":values,"cold_us":elapsed,"warm_us":start.elapsed().as_micros(),"cold_reads":cold_reads,"warm_reads":store.reads.load(Ordering::Relaxed)-cold_reads,"transport_bytes":store.bytes.load(Ordering::Relaxed),"retained_cache_bytes":cache.lock().unwrap().used(),"metric":reader.manifest.metric})
    );
    Ok(())
}
