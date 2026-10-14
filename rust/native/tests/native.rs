use serde::Deserialize;
use shakmaty::{CastlingMode, Chess, fen::Fen};
use shakmaty_syzygy::{
    Tablebase,
    filesystem::{Filesystem, ReadHint},
};
use std::{
    path::PathBuf,
    sync::atomic::Ordering,
    sync::{Arc, Mutex},
};
use tablebase_core::{
    Result,
    archive::{FileStore, RangeStore, SharedCache},
    range_cache::TileStore,
};
use tablebase_native::{Catalog, FilesystemHandle, NativeFilesystem};
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}
#[derive(Deserialize)]
struct Queries {
    fens: Vec<String>,
    values: Vec<i32>,
    dtz: Vec<i32>,
}
fn setup(budget: usize) -> (Tablebase<Chess>, Arc<NativeFilesystem>, Arc<FileStore>) {
    let catalog =
        Catalog::parse(&std::fs::read(root().join("native-catalog.json")).unwrap()).unwrap();
    let origin = Arc::new(FileStore::new(root()));
    let store = Arc::new(
        TileStore::new(
            origin.clone(),
            catalog.object_sizes(),
            16384,
            Arc::new(Mutex::new(SharedCache::new(0))),
        )
        .unwrap(),
    );
    let filesystem = Arc::new(NativeFilesystem::new(catalog, store, budget, 1 << 20));
    let mut tables = Tablebase::with_filesystem(Arc::new(FilesystemHandle(filesystem.clone())));
    for name in filesystem.files.keys() {
        tables.add_file(name).unwrap();
    }
    (tables, filesystem, origin)
}
#[test]
fn native_wdl_dtz_batches_and_eviction() {
    let query: Queries =
        serde_json::from_slice(&std::fs::read(root().join("queries.json")).unwrap()).unwrap();
    let positions: Vec<Chess> = query
        .fens
        .iter()
        .map(|f| {
            f.parse::<Fen>()
                .unwrap()
                .into_position(CastlingMode::Standard)
                .unwrap()
        })
        .collect();
    for budget in [0, 1024, 1 << 20] {
        let (tables, filesystem, origin) = setup(budget);
        let (values, _, _) = filesystem.probe_many(&tables, &positions, 32).unwrap();
        assert_eq!(values, query.values);
        for ((p, &w), &d) in positions.iter().zip(&query.values).zip(&query.dtz) {
            assert_eq!(tables.probe_wdl_after_zeroing(p).unwrap() as i32, w);
            assert_eq!(tables.probe_dtz(p).unwrap().ignore_rounding().0, d);
        }
        let reads = origin.reads.load(Ordering::Relaxed);
        assert_eq!(
            filesystem.probe_many(&tables, &positions, 32).unwrap().0,
            query.values
        );
        if budget == 1 << 20 {
            assert_eq!(origin.reads.load(Ordering::Relaxed), reads);
        }
        assert!(filesystem.retained_bytes().unwrap().0 <= budget);
        assert!(filesystem.probe_many(&tables, &positions, 0).is_err());
        assert!(
            filesystem
                .probe_many(&tables, &[], 32)
                .unwrap()
                .0
                .is_empty()
        );
        let missing: Chess = "8/8/8/8/8/7k/4r3/KQ6 w - - 0 1"
            .parse::<Fen>()
            .unwrap()
            .into_position(CastlingMode::Standard)
            .unwrap();
        assert!(filesystem.probe_many(&tables, &[missing], 32).is_err());
    }
}
struct Fault {
    origin: FileStore,
    key: String,
    short: bool,
}
#[test]
fn concurrent_native_batch_sessions() {
    let query: Queries =
        serde_json::from_slice(&std::fs::read(root().join("queries.json")).unwrap()).unwrap();
    let positions: Vec<Chess> = query
        .fens
        .iter()
        .map(|f| {
            f.parse::<Fen>()
                .unwrap()
                .into_position(CastlingMode::Standard)
                .unwrap()
        })
        .collect();
    for budget in [1024, 1 << 20] {
        let (tables, filesystem, _) = setup(budget);
        let tables = Arc::new(tables);
        std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|offset| {
                    let tables = tables.clone();
                    let filesystem = filesystem.clone();
                    let positions = &positions;
                    let expected = &query.values;
                    scope.spawn(move || {
                        let mut rotated = positions.clone();
                        let mut values = expected.clone();
                        rotated.rotate_left(offset);
                        values.rotate_left(offset);
                        assert_eq!(
                            filesystem.probe_many(&tables, &rotated, 16).unwrap().0,
                            values
                        );
                    })
                })
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
        });
        assert!(filesystem.retained_bytes().unwrap().0 <= budget);
    }
}
impl RangeStore for Fault {
    fn read(&self, key: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        let mut data = self.origin.read(key, offset, length)?;
        if self.short {
            data.pop();
        } else if key == self.key && !data.is_empty() {
            data[0] ^= 1;
        }
        Ok(data)
    }
}
#[test]
fn authenticated_native_bytes_and_catalog() {
    let raw = std::fs::read(root().join("native-catalog.json")).unwrap();
    let c = Catalog::parse(&raw).unwrap();
    let file = c.files.iter().find(|f| f.name == "KQvK.rtbw").unwrap();
    for key in [&file.key, &file.index_key] {
        let filesystem = Arc::new(NativeFilesystem::new(
            Catalog::parse(&raw).unwrap(),
            Arc::new(Fault {
                origin: FileStore::new(root()),
                key: key.clone(),
                short: false,
            }),
            1 << 20,
            1 << 20,
        ));
        let handle = FilesystemHandle(filesystem);
        let file = handle.open(std::path::Path::new("KQvK.rtbw")).unwrap();
        assert!(file.read_at(&mut [0u8; 8], 0, ReadHint::Header).is_err());
    }
    let filesystem = Arc::new(NativeFilesystem::new(
        Catalog::parse(&raw).unwrap(),
        Arc::new(Fault {
            origin: FileStore::new(root()),
            key: String::new(),
            short: true,
        }),
        1 << 20,
        1 << 20,
    ));
    assert!(
        FilesystemHandle(filesystem)
            .open(std::path::Path::new("KQvK.rtbw"))
            .unwrap()
            .read_at(&mut [0u8; 8], 0, ReadHint::Header)
            .is_err()
    );
    let mut invalid: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    invalid["files"][0]["size"] = serde_json::json!(1);
    assert!(Catalog::parse(invalid.to_string().as_bytes()).is_err());
}
