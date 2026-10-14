use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    sync::{Arc, Barrier, Mutex},
    time::Duration,
};
use tablebase_core::{
    Result,
    archive::{RangeStore, SharedCache},
    range_cache::TileStore,
};
struct Slow {
    calls: AtomicU64,
    fail: AtomicBool,
    panic: AtomicBool,
}
impl RangeStore for Slow {
    fn read(&self, _: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            !self.panic.load(Ordering::Relaxed),
            "injected transport panic"
        );
        if self.fail.load(Ordering::Relaxed) {
            return Err("injected failure".into());
        }
        Ok((0..length).map(|i| (offset + i as u64) as u8).collect())
    }
}
#[test]
fn concurrent_join_cache_failure_and_panic_recovery() {
    let origin = Arc::new(Slow {
        calls: AtomicU64::new(0),
        fail: AtomicBool::new(false),
        panic: AtomicBool::new(false),
    });
    let cache = Arc::new(Mutex::new(SharedCache::new(4096)));
    let store = Arc::new(
        TileStore::new(
            origin.clone(),
            BTreeMap::from([("object".into(), 4096)]),
            256,
            cache,
        )
        .unwrap(),
    );
    let gate = Arc::new(Barrier::new(16));
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let gate = gate.clone();
                let store = store.clone();
                scope.spawn(move || {
                    gate.wait();
                    store.read("object", 7, 128).unwrap()
                })
            })
            .collect();
        for h in handles {
            assert_eq!(
                h.join().unwrap(),
                (7..135).map(|i| i as u8).collect::<Vec<_>>()
            );
        }
    });
    assert_eq!(origin.calls.load(Ordering::Relaxed), 1);
    assert_eq!(
        store.read("object", 15, 32).unwrap(),
        (15..47).map(|i| i as u8).collect::<Vec<_>>()
    );
    assert_eq!(origin.calls.load(Ordering::Relaxed), 1);
    origin.fail.store(true, Ordering::Relaxed);
    assert!(store.read("object", 512, 8).is_err());
    origin.fail.store(false, Ordering::Relaxed);
    assert_eq!(store.read("object", 512, 8).unwrap().len(), 8);
    origin.panic.store(true, Ordering::Relaxed);
    assert!(store.read("object", 1024, 8).is_err());
    origin.panic.store(false, Ordering::Relaxed);
    assert_eq!(store.read("object", 1024, 8).unwrap().len(), 8);
    assert!(store.read("object", 4095, 2).is_err());
    assert!(store.read("missing", 0, 1).is_err());
    assert!(store.read("object", u64::MAX, 2).is_err());
    assert!(store.read("object", 0, (1 << 20) + 1).is_err());
}
