//! Native Syzygy compression, exact FEN probing and authenticated physical ranges.
pub mod transport;
use lru::LruCache;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use shakmaty::Chess;
use shakmaty_syzygy::{
    Tablebase,
    filesystem::{Filesystem, RandomAccessFile, ReadHint},
};
use std::cell::RefCell;
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
};
use tablebase_core::{Result, archive::RangeStore};

fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}
fn hash_key(x: &str) -> bool {
    x.len() == 70
        && x.ends_with(".tbobj")
        && x[..64]
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
#[derive(Clone, Deserialize)]
pub struct NativeFile {
    pub name: String,
    pub key: String,
    pub size: u64,
    pub index_key: String,
    pub index_bytes: u64,
    pub hash_pages: Vec<String>,
}
#[derive(Deserialize)]
pub struct Catalog {
    pub format: String,
    pub tile_bytes: usize,
    pub hash_page_tiles: usize,
    pub files: Vec<NativeFile>,
}
impl Catalog {
    pub fn parse(raw: &[u8]) -> Result<Self> {
        let catalog: Self = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
        if catalog.format != "native-syzygy-ranges-v1"
            || catalog.tile_bytes != 16384
            || catalog.hash_page_tiles != 256
        {
            return Err("native catalog version/geometry".into());
        }
        let mut names = std::collections::BTreeSet::new();
        for f in &catalog.files {
            let valid_name = f
                .name
                .strip_suffix(".rtbw")
                .or_else(|| f.name.strip_suffix(".rtbz"))
                .is_some_and(|stem| stem.bytes().all(|b| b"KQRBNPv".contains(&b)));
            if !valid_name
                || !names.insert(&f.name)
                || !hash_key(&f.key)
                || !hash_key(&f.index_key)
                || f.size % 64 != 16
                || f.size > i64::MAX as u64
                || f.index_bytes
                    != f.size
                        .div_ceil(16384)
                        .checked_mul(32)
                        .ok_or("native index overflow")?
                || f.hash_pages.len() as u64 != f.index_bytes.div_ceil(8192)
                || f.hash_pages.iter().any(|h| {
                    h.len() != 64
                        || !h
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                })
            {
                return Err("native catalog file geometry".into());
            }
        }
        Ok(catalog)
    }
    pub fn object_sizes(&self) -> BTreeMap<String, u64> {
        self.files
            .iter()
            .flat_map(|f| {
                [
                    (f.key.clone(), f.size),
                    (f.index_key.clone(), f.index_bytes),
                ]
            })
            .collect()
    }
}
type TileKey = (String, u64);
thread_local! {static DEFERRED:RefCell<Option<std::collections::BTreeSet<TileKey>>>=const {RefCell::new(None)};}
struct DeferGuard;
impl Drop for DeferGuard {
    fn drop(&mut self) {
        DEFERRED.with(|x| *x.borrow_mut() = None);
    }
}
struct Cache {
    budget: usize,
    used: usize,
    values: LruCache<TileKey, Arc<Vec<u8>>>,
}
impl Cache {
    fn new(budget: usize) -> Self {
        Self {
            budget,
            used: 0,
            values: LruCache::unbounded(),
        }
    }
    fn get(&mut self, key: &TileKey) -> Option<Arc<Vec<u8>>> {
        self.values.get(key).cloned()
    }
    fn put(&mut self, key: TileKey, value: Arc<Vec<u8>>) {
        let size = value.len() + key.0.len() + 128;
        if size > self.budget {
            return;
        }
        if let Some(old) = self.values.pop(&key) {
            self.used -= old.len() + key.0.len() + 128;
        }
        while self.used + size > self.budget {
            if let Some((k, v)) = self.values.pop_lru() {
                self.used -= v.len() + k.0.len() + 128;
            } else {
                break;
            }
        }
        self.used += size;
        self.values.put(key, value);
    }
}
pub struct NativeFilesystem {
    pub files: BTreeMap<String, NativeFile>,
    store: Arc<dyn RangeStore>,
    verified: Mutex<Cache>,
    hash_pages: Mutex<Cache>,
    pub logical_reads: AtomicU64,
    pub read_hints: [AtomicU64; 5],
}
impl NativeFilesystem {
    pub fn new(
        catalog: Catalog,
        store: Arc<dyn RangeStore>,
        data_budget: usize,
        index_budget: usize,
    ) -> Self {
        Self {
            files: catalog
                .files
                .into_iter()
                .map(|f| (f.name.clone(), f))
                .collect(),
            store,
            verified: Mutex::new(Cache::new(data_budget)),
            hash_pages: Mutex::new(Cache::new(index_budget)),
            logical_reads: AtomicU64::new(0),
            read_hints: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
    fn tile(&self, file: &NativeFile, tile: u64) -> Result<Arc<Vec<u8>>> {
        let key = (file.key.clone(), tile);
        if let Some(bytes) = self
            .verified
            .lock()
            .map_err(|_| "native cache poisoned")?
            .get(&key)
        {
            return Ok(bytes);
        }
        let deferred = DEFERRED.with(|pending| {
            let mut pending = pending.borrow_mut();
            if let Some(pending) = pending.as_mut() {
                pending.insert((file.name.clone(), tile));
                true
            } else {
                false
            }
        });
        if deferred {
            return Err("deferred native tile".into());
        }
        let hashes = self.hashes(file, tile / 256)?;
        let begin = tile * 16384;
        let count = (file.size - begin).min(16384) as usize;
        let bytes = self.store.read(&file.key, begin, count)?;
        self.authenticate(file, tile, &bytes, &hashes)?;
        let bytes = Arc::new(bytes);
        self.verified
            .lock()
            .map_err(|_| "native cache poisoned")?
            .put(key, bytes.clone());
        Ok(bytes)
    }
    fn hashes(&self, file: &NativeFile, page: u64) -> Result<Arc<Vec<u8>>> {
        let hash_key = (file.index_key.clone(), page);
        let cached = self
            .hash_pages
            .lock()
            .map_err(|_| "native index cache poisoned")?
            .get(&hash_key);
        let hashes = if let Some(bytes) = cached {
            bytes
        } else {
            let offset = page * 8192;
            let count = (file.index_bytes - offset).min(8192) as usize;
            let bytes = self.store.read(&file.index_key, offset, count)?;
            if bytes.len() != count
                || hex(&Sha256::digest(&bytes)) != file.hash_pages[page as usize]
            {
                return Err("native hash-page checksum".into());
            }
            let bytes = Arc::new(bytes);
            self.hash_pages
                .lock()
                .map_err(|_| "native index cache poisoned")?
                .put(hash_key, bytes.clone());
            bytes
        };
        Ok(hashes)
    }
    fn authenticate(
        &self,
        file: &NativeFile,
        tile: u64,
        bytes: &[u8],
        hashes: &[u8],
    ) -> Result<()> {
        let count = (file.size - tile * 16384).min(16384) as usize;
        if bytes.len() != count
            || Sha256::digest(bytes).as_slice()
                != &hashes[(tile % 256) as usize * 32..(tile % 256 + 1) as usize * 32]
        {
            return Err("native data-tile checksum".into());
        }
        Ok(())
    }
    fn prefetch(&self, requests: std::collections::BTreeSet<TileKey>) -> Result<()> {
        let mut groups: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for (name, tile) in requests {
            let file = self.files.get(&name).ok_or("deferred file absent")?;
            if self
                .verified
                .lock()
                .map_err(|_| "cache poisoned")?
                .get(&(file.key.clone(), tile))
                .is_none()
            {
                groups.entry(name).or_default().push(tile);
            }
        }
        for (name, tiles) in groups {
            let file = &self.files[&name];
            let mut begin = 0;
            while begin < tiles.len() {
                let mut end = begin + 1;
                while end < tiles.len() && tiles[end] == tiles[end - 1] + 1 && end - begin < 64 {
                    end += 1;
                }
                let offset = tiles[begin] * 16384;
                let stop = ((tiles[end - 1] + 1) * 16384).min(file.size);
                let raw = self
                    .store
                    .read_coalesced(&file.key, offset, (stop - offset) as usize)?;
                if raw.len() as u64 != stop - offset {
                    return Err("short native prefetch".into());
                }
                for &tile in &tiles[begin..end] {
                    let hashes = self.hashes(file, tile / 256)?;
                    let at = (tile * 16384 - offset) as usize;
                    let n = (file.size - tile * 16384).min(16384) as usize;
                    let bytes = &raw[at..at + n];
                    self.authenticate(file, tile, bytes, &hashes)?;
                    self.verified
                        .lock()
                        .map_err(|_| "cache poisoned")?
                        .put((file.key.clone(), tile), Arc::new(bytes.to_vec()));
                }
                begin = end;
            }
        }
        Ok(())
    }
    /// Exact retrying batch probe. Failed reads become misses only if this
    /// thread's recorder contains a concrete tile; other probe errors propagate.
    /// Tiny caches fall back to exact scalar reads after bounded dependency waves.
    pub fn probe_many(
        &self,
        tables: &Tablebase<Chess>,
        positions: &[Chess],
        window: usize,
    ) -> Result<(Vec<i32>, usize, usize)> {
        if window == 0 || window > 4096 {
            return Err("native batch window must be 1..4096".into());
        }
        let mut output = vec![0; positions.len()];
        let mut waves = 0;
        let mut fallbacks = 0;
        let budget = self.verified.lock().map_err(|_| "cache poisoned")?.budget;
        if budget == 0 {
            for (index, position) in positions.iter().enumerate() {
                output[index] = tables
                    .probe_wdl_after_zeroing(position)
                    .map_err(|e| e.to_string())? as i32;
            }
            return Ok((output, 0, positions.len()));
        }
        let window = window.min((budget / (4 * 16384 + 1024)).max(1));
        for begin in (0..positions.len()).step_by(window) {
            let end = (begin + window).min(positions.len());
            let mut pending: Vec<_> = (begin..end).collect();
            let mut seen_requests = std::collections::BTreeSet::new();
            for _ in 0..16 {
                if pending.is_empty() {
                    break;
                }
                let mut next = Vec::new();
                let mut requests = std::collections::BTreeSet::new();
                for &index in &pending {
                    DEFERRED.with(|x| *x.borrow_mut() = Some(std::collections::BTreeSet::new()));
                    let guard = DeferGuard;
                    let value = tables.probe_wdl_after_zeroing(&positions[index]);
                    let missed = DEFERRED.with(|x| x.borrow_mut().take().unwrap());
                    drop(guard);
                    match value {
                        Ok(value) => output[index] = value as i32,
                        Err(error) => {
                            if missed.is_empty() {
                                return Err(error.to_string());
                            }
                            requests.extend(missed);
                            next.push(index);
                        }
                    }
                }
                pending = next;
                if !requests.is_empty() {
                    if !seen_requests.insert(requests.iter().cloned().collect::<Vec<_>>()) {
                        break;
                    }
                    self.prefetch(requests)?;
                    waves += 1;
                }
            }
            for index in pending {
                output[index] = tables
                    .probe_wdl_after_zeroing(&positions[index])
                    .map_err(|e| e.to_string())? as i32;
                fallbacks += 1;
            }
        }
        Ok((output, waves, fallbacks))
    }
    fn get_file(&self, path: &Path) -> io::Result<NativeFile> {
        path.file_name()
            .and_then(|x| x.to_str())
            .and_then(|name| self.files.get(name))
            .cloned()
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }
    pub fn retained_bytes(&self) -> Result<(usize, usize)> {
        Ok((
            self.verified.lock().map_err(|_| "cache poisoned")?.used,
            self.hash_pages.lock().map_err(|_| "cache poisoned")?.used,
        ))
    }
}
// Arc-owned filesystem handles keep immutable catalog and caches alive.
pub struct FilesystemHandle(pub Arc<NativeFilesystem>);
impl Filesystem for FilesystemHandle {
    fn regular_file_size(&self, path: &Path) -> io::Result<u64> {
        Ok(self.0.get_file(path)?.size)
    }
    fn read_dir(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        Ok(self.0.files.keys().map(|name| path.join(name)).collect())
    }
    fn open(&self, path: &Path) -> io::Result<Box<dyn RandomAccessFile>> {
        Ok(Box::new(NativeHandle {
            file: self.0.get_file(path)?,
            filesystem: self.0.clone(),
        }))
    }
}
struct NativeHandle {
    file: NativeFile,
    filesystem: Arc<NativeFilesystem>,
}
impl RandomAccessFile for NativeHandle {
    fn read_at(&self, buffer: &mut [u8], offset: u64, hint: ReadHint) -> io::Result<usize> {
        self.filesystem
            .logical_reads
            .fetch_add(1, Ordering::Relaxed);
        let hint = match hint {
            ReadHint::Header => 0,
            ReadHint::DtzMap => 1,
            ReadHint::SparseIndex => 2,
            ReadHint::BlockLengths => 3,
            ReadHint::Data => 4,
        };
        self.filesystem.read_hints[hint].fetch_add(1, Ordering::Relaxed);
        if offset >= self.file.size {
            return Ok(0);
        }
        let count = (self.file.size - offset).min(buffer.len() as u64) as usize;
        let mut copied = 0;
        while copied < count {
            let position = offset + copied as u64;
            let tile = position / 16384;
            let bytes = self
                .filesystem
                .tile(&self.file, tile)
                .map_err(io::Error::other)?;
            let begin = (position % 16384) as usize;
            let n = (bytes.len() - begin).min(count - copied);
            buffer[copied..copied + n].copy_from_slice(&bytes[begin..begin + n]);
            copied += n;
        }
        Ok(count)
    }
}
