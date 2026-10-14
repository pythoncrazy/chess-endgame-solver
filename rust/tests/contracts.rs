use serde::Deserialize;
use std::{
    path::PathBuf,
    sync::atomic::Ordering,
    sync::{Arc, Mutex},
};
use tablebase_core::{
    archive::{FileStore, Manifest, RangeStore, Reader, SharedCache},
    codec::{Page, hash128},
    encode::encode,
    retrograde::{Wdl, solve, update_row},
};
struct FaultStore {
    origin: FileStore,
    corrupt_at: Option<u64>,
    short: bool,
}
impl RangeStore for FaultStore {
    fn read(&self, key: &str, offset: u64, length: usize) -> tablebase_core::Result<Vec<u8>> {
        let mut bytes = self.origin.read(key, offset, length)?;
        if let Some(at) = self.corrupt_at
            && at >= offset
            && at < offset + length as u64
        {
            bytes[(at - offset) as usize] ^= 1;
        }
        if self.short {
            bytes.pop();
        }
        Ok(bytes)
    }
}
#[test]
fn authenticated_directory_index_payload_and_short_reads() {
    let directory = root().join("fixtures/mixed");
    let raw = std::fs::read(directory.join("manifest.json")).unwrap();
    let manifest = Manifest::from_json(&raw).unwrap();
    let o = &manifest.objects[0];
    let positions: Vec<_> = (o.start..o.start + o.count).collect();
    let points = [0, 36 + o.directory_bytes + 3, o.bytes - 1];
    for at in points {
        let store = Arc::new(FaultStore {
            origin: FileStore::new(&directory),
            corrupt_at: Some(at),
            short: false,
        });
        let r = Reader::new(
            Manifest::from_json(&raw).unwrap(),
            store,
            Arc::new(Mutex::new(SharedCache::new(0))),
        );
        assert!(
            r.probe_many(&positions).is_err(),
            "corruption accepted at {at}"
        );
    }
    let r = Reader::new(
        Manifest::from_json(&raw).unwrap(),
        Arc::new(FaultStore {
            origin: FileStore::new(&directory),
            corrupt_at: None,
            short: true,
        }),
        Arc::new(Mutex::new(SharedCache::new(0))),
    );
    assert!(r.probe_many(&positions).is_err());
}
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn unhex(x: &str) -> Vec<u8> {
    (0..x.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&x[i..i + 2], 16).unwrap())
        .collect()
}
#[derive(Deserialize)]
struct Fixture {
    frame: String,
    hash: String,
    values: Vec<i64>,
    prediction: Option<Vec<i64>>,
}
#[test]
fn independent_codec_fixtures() {
    let fixtures: Vec<Fixture> =
        serde_json::from_slice(&std::fs::read(root().join("fixtures/frames.json")).unwrap())
            .unwrap();
    let mut modes = [false; 10];
    let mut compressors = [false; 3];
    for f in &fixtures {
        let raw = unhex(&f.frame);
        let expected: [u8; 16] = unhex(&f.hash).try_into().unwrap();
        modes[raw[4] as usize] = true;
        compressors[raw[5] as usize] = true;
        for authenticated in [None, Some(expected)] {
            let page = Page::parse(&raw, authenticated, f.prediction.as_deref()).unwrap();
            assert_eq!(page.n, f.values.len());
            for (i, &value) in f.values.iter().enumerate() {
                assert_eq!(
                    page.get(i).unwrap(),
                    value,
                    "mode {} comp {}",
                    raw[4],
                    raw[5]
                );
            }
            assert!(page.get(page.n).is_err());
        }
        for at in [0, 4, 6, 7, 8, 12, 16, raw.len() - 1] {
            let mut corrupt = raw.clone();
            corrupt[at] ^= 1;
            assert!(Page::parse(&corrupt, Some(expected), f.prediction.as_deref()).is_err());
        }
        let mut appended = raw.clone();
        appended.push(0);
        assert!(Page::parse(&appended, None, f.prediction.as_deref()).is_err());
        let encoded = encode(&f.values, 3).unwrap();
        let page = Page::parse(&encoded, None, None).unwrap();
        for (i, &value) in f.values.iter().enumerate() {
            assert_eq!(page.get(i).unwrap(), value);
        }
        if [4, 5].contains(&raw[4]) {
            assert!(Page::parse(&raw, None, None).is_err());
            let mut wrong = f.prediction.clone().unwrap();
            wrong[0] += 1;
            assert!(Page::parse(&raw, None, Some(&wrong)).is_err());
        }
    }
    assert!(modes.iter().all(|&x| x));
    assert!(compressors.iter().all(|&x| x));
    eprintln!(
        "independent frames={} mutated_frames={}",
        fixtures.len(),
        fixtures.len() * 8
    );
}
#[derive(Deserialize)]
struct ArchiveFixture {
    directory: String,
    values: Vec<i64>,
}
fn reader(directory: &std::path::Path, cache: Arc<Mutex<SharedCache>>) -> (Reader, Arc<FileStore>) {
    let m = Manifest::from_json(&std::fs::read(directory.join("manifest.json")).unwrap()).unwrap();
    let store = Arc::new(FileStore::new(directory));
    (Reader::new(m, store.clone(), cache), store)
}
#[test]
fn shared_cache_small_archives_and_eviction() {
    let fixtures: Vec<ArchiveFixture> =
        serde_json::from_slice(&std::fs::read(root().join("fixtures/archives.json")).unwrap())
            .unwrap();
    for budget in [0, 1024, 1 << 20] {
        let cache = Arc::new(Mutex::new(SharedCache::new(budget)));
        for f in &fixtures {
            let (mut r, _) = reader(&root().join("fixtures").join(&f.directory), cache.clone());
            r.max_range = 64;
            r.working_budget = 65 * 8;
            let ids: Vec<_> = (0..f.values.len() as u64)
                .rev()
                .chain([0, 0, 2, 2])
                .collect();
            let expected: Vec<_> = ids.iter().map(|&i| f.values[i as usize]).collect();
            assert_eq!(r.probe_many(&ids).unwrap(), expected);
            assert_eq!(r.probe_many(&ids).unwrap(), expected);
            assert!(r.probe_many(&[f.values.len() as u64]).is_err());
            assert!(r.probe_many(&[]).unwrap().is_empty());
            assert!(cache.lock().unwrap().used() <= budget);
        }
    }
}
#[derive(Deserialize)]
struct Queries {
    ids: Vec<u64>,
    values: Vec<i64>,
}
#[test]
fn complete_chess_archive_native_ids() {
    let queries: Queries =
        serde_json::from_slice(&std::fs::read(root().join("fixtures/chess_queries.json")).unwrap())
            .unwrap();
    let (r, store) = reader(
        &root().join("../bridge_final_example"),
        Arc::new(Mutex::new(SharedCache::new(64 << 20))),
    );
    assert_eq!(r.probe_many(&queries.ids).unwrap(), queries.values);
    let reads = store.reads.load(Ordering::Relaxed);
    assert_eq!(r.probe_many(&queries.ids).unwrap(), queries.values);
    assert_eq!(store.reads.load(Ordering::Relaxed), reads);
    let (legacy, _) = reader(
        &root().join("../bridge_fast_example"),
        Arc::new(Mutex::new(SharedCache::new(64 << 20))),
    );
    assert_eq!(legacy.probe_many(&queries.ids).unwrap(), queries.values);
}
#[test]
fn decimal_ids_and_manifest_rejections() {
    let root = serde_json::json!({"format":"exact-blocks-v3","metric":"exact","positions":"9007199254740999","page_positions":65,"index_page_records":2,"record_bytes":32,"constant":"-9223372036854775808","objects":[]});
    let m = Manifest::from_json(root.to_string().as_bytes()).unwrap();
    let r = Reader::new(
        m,
        Arc::new(FileStore::new("/nonexistent")),
        Arc::new(Mutex::new(SharedCache::new(0))),
    );
    assert_eq!(r.probe_many(&[9007199254740998]).unwrap(), vec![i64::MIN]);
    for (key, value) in [
        ("positions", serde_json::json!(1.5)),
        ("positions", serde_json::json!(true)),
        ("positions", serde_json::json!("01")),
        ("record_bytes", serde_json::json!(16)),
        ("page_positions", serde_json::json!(0)),
    ] {
        let mut bad = root.clone();
        bad[key] = value;
        assert!(Manifest::from_json(bad.to_string().as_bytes()).is_err());
    }
}
#[test]
fn retrograde_cycles_terminals_and_duplicates() {
    let edges: Vec<_> = (0..257)
        .map(|i| if i < 256 { vec![i + 1] } else { vec![] })
        .collect();
    let mut seeds = vec![None; 257];
    seeds[256] = Some(Wdl::Loss);
    let solved = solve(&edges, &seeds).unwrap();
    for (i, value) in solved.into_iter().enumerate() {
        assert_eq!(value, if i % 2 == 0 { Wdl::Loss } else { Wdl::Win });
    }
    assert_eq!(
        solve(
            &[vec![], vec![0, 0], vec![1], vec![3]],
            &[Some(Wdl::Loss), None, None, None]
        )
        .unwrap(),
        vec![Wdl::Loss, Wdl::Win, Wdl::Loss, Wdl::Draw]
    );
    assert_eq!(
        solve(&[vec![], vec![0]], &[Some(Wdl::Draw), None]).unwrap(),
        vec![Wdl::Draw, Wdl::Draw]
    );
    assert!(solve(&[vec![]], &[None]).is_err());
    assert!(solve(&[vec![1]], &[None]).is_err());
    for legal in 0..256u64 {
        for wins in 0..256u64 {
            let expected = if legal != 0 && wins & legal == legal {
                Some(Wdl::Loss)
            } else {
                None
            };
            assert_eq!(update_row(true, 0, wins, legal), expected);
        }
    }
    assert_eq!(update_row(true, 1, 0, 1), Some(Wdl::Win));
    assert_eq!(update_row(false, 1, 0, 1), None);
    assert_ne!(hash128(b"a"), hash128(b"b"));
}
#[derive(Deserialize)]
struct Graph {
    edges: Vec<Vec<usize>>,
    seeds: Vec<Option<i8>>,
    values: Vec<i8>,
}
#[test]
fn independent_graph_fixtures() {
    let graphs: Vec<Graph> =
        serde_json::from_slice(&std::fs::read(root().join("fixtures/graph_inputs.json")).unwrap())
            .unwrap();
    fn wdl(value: i8) -> Wdl {
        match value {
            -1 => Wdl::Loss,
            0 => Wdl::Draw,
            1 => Wdl::Win,
            _ => panic!("invalid fixture"),
        }
    }
    for graph in graphs {
        let seeds: Vec<_> = graph.seeds.into_iter().map(|v| v.map(wdl)).collect();
        assert_eq!(
            solve(&graph.edges, &seeds).unwrap(),
            graph.values.into_iter().map(wdl).collect::<Vec<_>>()
        );
    }
}
