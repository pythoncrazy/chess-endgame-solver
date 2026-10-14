//! Stable physical tiles for reusable origin requests and concurrent single-flight.
use crate::{
    Result,
    archive::{CacheKey, Cached, RangeStore, SharedCache},
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Condvar, Mutex},
};
type FlightResult = Result<Arc<Vec<u8>>>;
struct Flight {
    result: Mutex<Option<FlightResult>>,
    ready: Condvar,
}
pub struct TileStore {
    origin: Arc<dyn RangeStore>,
    sizes: BTreeMap<String, u64>,
    tile: usize,
    cache: Arc<Mutex<SharedCache>>,
    flights: Mutex<HashMap<CacheKey, Arc<Flight>>>,
    pub origin_reads: AtomicU64,
    pub joined_reads: AtomicU64,
    pub cache_hits: AtomicU64,
}
impl TileStore {
    pub fn new(
        origin: Arc<dyn RangeStore>,
        sizes: BTreeMap<String, u64>,
        tile: usize,
        cache: Arc<Mutex<SharedCache>>,
    ) -> Result<Self> {
        if tile == 0 || tile > 1 << 20 {
            return Err("tile size must be 1..1MiB".into());
        }
        Ok(Self {
            origin,
            sizes,
            tile,
            cache,
            flights: Mutex::new(HashMap::new()),
            origin_reads: AtomicU64::new(0),
            joined_reads: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
        })
    }
    fn load(&self, key: CacheKey) -> FlightResult {
        // Consistent lock order: flights, then cache. No I/O under either lock.
        let mut flights = self.flights.lock().map_err(|_| "flights poisoned")?;
        if let Some(value) = self.cache.lock().map_err(|_| "cache poisoned")?.get(&key) {
            let Cached::Raw(bytes) = &*value else {
                return Err("range cache kind".into());
            };
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(bytes.clone());
        }
        if let Some(flight) = flights.get(&key).cloned() {
            self.joined_reads.fetch_add(1, Ordering::Relaxed);
            drop(flights);
            let mut result = flight.result.lock().map_err(|_| "flight poisoned")?;
            while result.is_none() {
                result = flight
                    .ready
                    .wait(result)
                    .map_err(|_| "flight wait poisoned")?;
            }
            return result.as_ref().unwrap().clone();
        }
        let flight = Arc::new(Flight {
            result: Mutex::new(None),
            ready: Condvar::new(),
        });
        flights.insert(key.clone(), flight.clone());
        drop(flights);
        self.origin_reads.fetch_add(1, Ordering::Relaxed);
        // Convert a transport panic into a shared failure so waiters cannot hang.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.origin.read(&key.object, key.offset, key.length)
        }))
        .map_err(|_| "range transport panicked".to_string())
        .and_then(|x| x)
        .and_then(|bytes| {
            if bytes.len() != key.length {
                Err("short tile read".into())
            } else {
                Ok(Arc::new(bytes))
            }
        });
        let result = match result {
            Ok(bytes) => match self.cache.lock() {
                Ok(mut cache) => {
                    cache.put(key.clone(), Arc::new(Cached::Raw(bytes.clone())));
                    Ok(bytes)
                }
                Err(_) => Err("cache poisoned".into()),
            },
            Err(error) => Err(error),
        };
        *flight.result.lock().map_err(|_| "flight poisoned")? = Some(result.clone());
        flight.ready.notify_all();
        self.flights
            .lock()
            .map_err(|_| "flights poisoned")?
            .remove(&key);
        result
    }
}
impl RangeStore for TileStore {
    fn read_coalesced(&self, key: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        let size = *self.sizes.get(key).ok_or("object absent from catalog")?;
        if length == 0
            || length > 1 << 20
            || offset.checked_add(length as u64).ok_or("range overflow")? > size
        {
            return Err("coalesced range bounds".into());
        }
        Ok((*self.load(CacheKey {
            object: key.into(),
            kind: 3,
            offset,
            length,
        })?)
        .clone())
    }
    fn read(&self, key: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        if length > 1 << 20 {
            return Err("request exceeds 1MiB".into());
        }
        let size = *self.sizes.get(key).ok_or("object absent from catalog")?;
        let end = offset.checked_add(length as u64).ok_or("range overflow")?;
        if end > size {
            return Err("range outside object".into());
        }
        let mut output = Vec::with_capacity(length);
        let mut at = offset;
        while at < end {
            let start = at / self.tile as u64 * self.tile as u64;
            let tile_len = (size - start).min(self.tile as u64) as usize;
            let bytes = self.load(CacheKey {
                object: key.to_string(),
                kind: 3,
                offset: start,
                length: tile_len,
            })?;
            let begin = (at - start) as usize;
            let n = (tile_len - begin).min((end - at) as usize);
            output.extend_from_slice(&bytes[begin..begin + n]);
            at += n as u64;
        }
        Ok(output)
    }
}
