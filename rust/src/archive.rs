//! Authenticated paged indexes, bounded shared caches, and capped range reads.
//! Catalogs are trusted roots. They must come from the intended publisher.
use crate::{
    Result,
    codec::{Page, slice, u32le, u64le},
};
use lru::LruCache;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    sync::{Arc, Mutex},
};

fn integer<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<u64, D::Error> {
    let value = serde_json::Value::deserialize(d)?;
    let result = match value {
        serde_json::Value::Number(x) => x.as_u64(),
        serde_json::Value::String(x) => {
            if x == "0" || (!x.starts_with('0') && x.bytes().all(|b| b.is_ascii_digit())) {
                x.parse().ok()
            } else {
                None
            }
        }
        _ => None,
    };
    result
        .filter(|&x| x <= i64::MAX as u64)
        .ok_or_else(|| serde::de::Error::custom("exact nonnegative integer required"))
}
fn signed<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<i64>, D::Error> {
    let value = serde_json::Value::deserialize(d)?;
    let parsed = match value {
        serde_json::Value::Number(x) => x.as_i64(),
        serde_json::Value::String(x) => {
            let digits = x.strip_prefix('-').unwrap_or(&x);
            if digits == "0"
                || (!digits.starts_with('0')
                    && !digits.is_empty()
                    && digits.bytes().all(|b| b.is_ascii_digit()))
            {
                x.parse().ok()
            } else {
                None
            }
        }
        _ => None,
    };
    parsed
        .map(Some)
        .ok_or_else(|| serde::de::Error::custom("signed integer required"))
}
#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub metric: String,
    #[serde(deserialize_with = "integer")]
    pub positions: u64,
    #[serde(deserialize_with = "integer")]
    pub page_positions: u64,
    #[serde(deserialize_with = "integer")]
    pub index_page_records: u64,
    #[serde(default = "legacy_record_bytes", deserialize_with = "integer")]
    pub record_bytes: u64,
    #[serde(default, deserialize_with = "signed")]
    pub constant: Option<i64>,
    pub objects: Vec<Object>,
}
fn legacy_record_bytes() -> u64 {
    16
}
#[derive(Debug, Deserialize)]
pub struct Object {
    pub key: String,
    #[serde(deserialize_with = "integer")]
    pub start: u64,
    #[serde(deserialize_with = "integer")]
    pub count: u64,
    #[serde(deserialize_with = "integer")]
    pub pages: u64,
    #[serde(deserialize_with = "integer")]
    pub index_bytes: u64,
    #[serde(deserialize_with = "integer")]
    pub directory_bytes: u64,
    #[serde(deserialize_with = "integer")]
    pub bytes: u64,
    pub directory_sha256: String,
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|x| format!("{x:02x}")).collect()
}
fn is_hash(x: &str) -> bool {
    x.len() == 64
        && x.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl Manifest {
    pub fn from_json(raw: &[u8]) -> Result<Self> {
        let m: Self = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
        if !["exact-blocks-v2", "exact-blocks-v3"].contains(&m.format.as_str())
            || m.metric.is_empty()
            || !(1..=1 << 20).contains(&m.page_positions)
            || !(1..=1 << 20).contains(&m.index_page_records)
            || m.record_bytes
                != if m.format == "exact-blocks-v3" {
                    32
                } else {
                    16
                }
        {
            return Err("manifest geometry/version".into());
        }
        if m.constant.is_some() {
            if !m.objects.is_empty() {
                return Err("constant objects".into());
            }
            return Ok(m);
        }
        let mut next = 0u64;
        for o in &m.objects {
            if !o.key.ends_with(".tbobj")
                || !is_hash(&o.key[..o.key.len() - 6])
                || !is_hash(&o.directory_sha256)
                || o.start != next
                || o.count == 0
                || o.pages > 1 << 20
                || o.pages != o.count.div_ceil(m.page_positions)
                || o.index_bytes
                    != o.pages
                        .checked_mul(m.record_bytes)
                        .ok_or("index overflow")?
                || o.directory_bytes
                    != o.pages
                        .div_ceil(m.index_page_records)
                        .checked_mul(32)
                        .ok_or("directory overflow")?
                || o.bytes
                    < 36u64
                        .checked_add(o.directory_bytes)
                        .and_then(|x| x.checked_add(o.index_bytes))
                        .ok_or("object overflow")?
            {
                return Err("invalid object coverage or identity".into());
            }
            next = next
                .checked_add(o.count)
                .filter(|x| *x <= i64::MAX as u64)
                .ok_or("ID overflow")?;
        }
        if next != m.positions {
            return Err("incomplete coverage".into());
        }
        Ok(m)
    }
}
pub trait RangeStore: Send + Sync {
    fn read(&self, key: &str, offset: u64, length: usize) -> Result<Vec<u8>>;
    /// Planner-requested contiguous read; implementations may preserve it as one GET.
    fn read_coalesced(&self, key: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        self.read(key, offset, length)
    }
}
pub struct FileStore {
    pub directory: PathBuf,
    pub reads: AtomicU64,
    pub bytes: AtomicU64,
}
impl FileStore {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
            reads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }
}
impl RangeStore for FileStore {
    fn read(&self, key: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        if !key.ends_with(".tbobj") || !is_hash(&key[..key.len() - 6]) {
            return Err("invalid object key".into());
        }
        let mut file = File::open(self.directory.join(key)).map_err(|e| e.to_string())?;
        let end = offset.checked_add(length as u64).ok_or("range overflow")?;
        if end > file.metadata().map_err(|e| e.to_string())?.len() {
            return Err("range beyond object".into());
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| e.to_string())?;
        let mut raw = vec![0; length];
        file.read_exact(&mut raw).map_err(|e| e.to_string())?;
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(length as u64, Ordering::Relaxed);
        Ok(raw)
    }
}
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct CacheKey {
    pub object: String,
    pub kind: u8,
    pub offset: u64,
    pub length: usize,
}
pub enum Cached {
    Raw(Arc<Vec<u8>>),
    Page(Page),
    Constant(i64),
}
impl Cached {
    fn bytes(&self) -> usize {
        match self {
            Self::Raw(x) => x.len(),
            Self::Page(x) => x.retained_bytes(),
            Self::Constant(_) => 8,
        }
    }
}
/// Global budget across materials. Charges payload, key, and 128 bytes per entry.
/// Allocator overhead and temporary clones/in-flight buffers are additional.
pub struct SharedCache {
    budget: usize,
    used: usize,
    items: LruCache<CacheKey, Arc<Cached>>,
}
impl SharedCache {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            used: 0,
            items: LruCache::unbounded(),
        }
    }
    pub fn used(&self) -> usize {
        self.used
    }
    fn cost(key: &CacheKey, value: &Cached) -> usize {
        value.bytes() + key.object.len() + 128
    }
    pub(crate) fn get(&mut self, key: &CacheKey) -> Option<Arc<Cached>> {
        self.items.get(key).cloned()
    }
    pub(crate) fn put(&mut self, key: CacheKey, value: Arc<Cached>) {
        let size = Self::cost(&key, &value);
        if size > self.budget {
            return;
        }
        if let Some(old) = self.items.pop(&key) {
            self.used -= Self::cost(&key, &old);
        }
        while self.used + size > self.budget {
            if let Some((k, v)) = self.items.pop_lru() {
                self.used -= Self::cost(&k, &v);
            } else {
                break;
            }
        }
        self.used += size;
        self.items.put(key, value);
    }
}
pub struct Reader {
    pub manifest: Manifest,
    pub store: Arc<dyn RangeStore>,
    pub cache: Arc<Mutex<SharedCache>>,
    pub max_range: usize,
    pub working_budget: usize,
}
struct Record {
    offset: u64,
    length: usize,
    constant: i64,
    checksum: Option<[u8; 16]>,
}
impl Reader {
    pub fn new(
        manifest: Manifest,
        store: Arc<dyn RangeStore>,
        cache: Arc<Mutex<SharedCache>>,
    ) -> Self {
        Self {
            manifest,
            store,
            cache,
            max_range: 1 << 20,
            working_budget: 64 << 20,
        }
    }
    fn cache_get(&self, key: &CacheKey) -> Result<Option<Arc<Cached>>> {
        Ok(self.cache.lock().map_err(|_| "cache poisoned")?.get(key))
    }
    fn cache_put(&self, key: CacheKey, value: Arc<Cached>) -> Result<()> {
        self.cache
            .lock()
            .map_err(|_| "cache poisoned")?
            .put(key, value);
        Ok(())
    }
    fn key(o: &Object, kind: u8, offset: u64, length: usize) -> CacheKey {
        CacheKey {
            object: o.key.clone(),
            kind,
            offset,
            length,
        }
    }
    fn fetch(&self, o: &Object, offset: u64, length: usize) -> Result<Vec<u8>> {
        if self.max_range == 0 || self.max_range > 1 << 20 {
            return Err("range cap must be 1..1MiB".into());
        }
        if offset.checked_add(length as u64).ok_or("range overflow")? > o.bytes {
            return Err("object range".into());
        }
        let mut raw = Vec::with_capacity(length);
        for begin in (0..length).step_by(self.max_range) {
            let n = (length - begin).min(self.max_range);
            let piece = self.store.read(&o.key, offset + begin as u64, n)?;
            if piece.len() != n {
                return Err("short range".into());
            }
            raw.extend_from_slice(&piece);
        }
        Ok(raw)
    }
    fn directory(&self, o: &Object) -> Result<Arc<Cached>> {
        let length =
            usize::try_from(36 + o.directory_bytes).map_err(|_| "directory addressability")?;
        let key = Self::key(o, 0, 0, length);
        if let Some(value) = self.cache_get(&key)? {
            return Ok(value);
        }
        let raw = self.fetch(o, 0, length)?;
        if hex(&Sha256::digest(&raw)) != o.directory_sha256 {
            return Err("directory checksum".into());
        }
        let version = if self.manifest.record_bytes == 32 {
            3
        } else {
            2
        };
        if &raw[..8]
            != if version == 3 {
                b"TBOBJ003"
            } else {
                b"TBOBJ002"
            }
            || u32le(&raw, 8)? != version
            || u32le(&raw, 12)? as u64 != self.manifest.page_positions
            || u64le(&raw, 16)? != o.start
            || u64le(&raw, 24)? != o.count
            || u32le(&raw, 32)? as u64 != o.pages
        {
            return Err("object header".into());
        }
        let result = Arc::new(Cached::Raw(Arc::new(raw)));
        self.cache_put(key, result.clone())?;
        Ok(result)
    }
    fn record(&self, o: &Object, page: u64) -> Result<Record> {
        let chunk = page / self.manifest.index_page_records;
        let chunk_start = chunk * self.manifest.index_page_records * self.manifest.record_bytes;
        let offset = 36 + o.directory_bytes + chunk_start;
        let length = (self.manifest.index_page_records * self.manifest.record_bytes)
            .min(o.index_bytes - chunk_start) as usize;
        let key = Self::key(o, 1, offset, length);
        let index = if let Some(value) = self.cache_get(&key)? {
            value
        } else {
            let directory = self.directory(o)?;
            let Cached::Raw(hashes) = &*directory else {
                return Err("cache kind".into());
            };
            let raw = self.fetch(o, offset, length)?;
            if Sha256::digest(&raw).as_slice() != slice(hashes, 36 + chunk as usize * 32, 32)? {
                return Err("index checksum".into());
            }
            let value = Arc::new(Cached::Raw(Arc::new(raw)));
            self.cache_put(key, value.clone())?;
            value
        };
        let Cached::Raw(raw) = &*index else {
            return Err("cache kind".into());
        };
        let at = ((page % self.manifest.index_page_records) * self.manifest.record_bytes) as usize;
        let record = Record {
            offset: u64le(raw, at)?,
            length: u32le(raw, at + 8)? as usize,
            constant: u32le(raw, at + 12)? as i32 as i64,
            checksum: if self.manifest.record_bytes == 32 {
                Some(slice(raw, at + 16, 16)?.try_into().unwrap())
            } else {
                None
            },
        };
        if record.length > (16 * self.manifest.page_positions + 65536) as usize {
            return Err("encoded frame exceeds resource cap".into());
        }
        if record.length > 0
            && (record.offset < 36 + o.directory_bytes + o.index_bytes
                || record
                    .offset
                    .checked_add(record.length as u64)
                    .ok_or("payload overflow")?
                    > o.bytes)
        {
            return Err("payload range".into());
        }
        Ok(record)
    }
    /// Sort/deduplicate pages. Process bounded windows, coalescing nearby payloads.
    /// The caller-owned output scales with batch size; no full archive is loaded.
    pub fn probe_many(&self, positions: &[u64]) -> Result<Vec<i64>> {
        if positions.iter().any(|&p| p >= self.manifest.positions) {
            return Err("position out of range".into());
        }
        if let Some(constant) = self.manifest.constant {
            return Ok(vec![constant; positions.len()]);
        }
        if self.working_budget == 0 {
            return Err("working budget".into());
        }
        let mut groups: BTreeMap<(usize, u64), Vec<(usize, usize)>> = BTreeMap::new();
        for (target, &position) in positions.iter().enumerate() {
            let j = self
                .manifest
                .objects
                .partition_point(|o| o.start <= position)
                - 1;
            let local = position - self.manifest.objects[j].start;
            groups
                .entry((j, local / self.manifest.page_positions))
                .or_default()
                .push((target, (local % self.manifest.page_positions) as usize));
        }
        let groups: Vec<_> = groups.into_iter().collect();
        let mut output = vec![0; positions.len()];
        let mut begin = 0;
        while begin < groups.len() {
            let owner = groups[begin].0.0;
            let o = &self.manifest.objects[owner];
            let window_pages =
                (self.working_budget / (self.manifest.page_positions as usize * 8)).max(1);
            let end = (begin + window_pages).min(groups.len());
            let end = (begin..end)
                .find(|&i| groups[i].0.0 != owner)
                .unwrap_or(end);
            let mut missing = Vec::new();
            for ((_, page), targets) in &groups[begin..end] {
                let key = Self::key(o, 2, *page, 0);
                if let Some(value) = self.cache_get(&key)? {
                    fill(&value, targets, &mut output)?;
                    continue;
                }
                let record = self.record(o, *page)?;
                if record.length == 0 {
                    let value = Arc::new(Cached::Constant(record.constant));
                    fill(&value, targets, &mut output)?;
                    self.cache_put(key, value)?;
                } else {
                    missing.push((*page, targets, record));
                }
            }
            missing.sort_by_key(|x| x.2.offset);
            let mut i = 0;
            while i < missing.len() {
                let offset = missing[i].2.offset;
                let mut range_end = offset + missing[i].2.length as u64;
                let mut useful = missing[i].2.length as u64;
                let mut stop = i + 1;
                while stop < missing.len() {
                    let next = &missing[stop].2;
                    let new_end = range_end.max(next.offset + next.length as u64);
                    if next.offset > range_end + 4096
                        || new_end - offset > self.max_range as u64
                        || new_end - offset > 2 * (useful + next.length as u64)
                    {
                        break;
                    }
                    range_end = new_end;
                    useful += next.length as u64;
                    stop += 1;
                }
                let raw = self.fetch(o, offset, (range_end - offset) as usize)?;
                for (page, targets, record) in &missing[i..stop] {
                    let frame = slice(&raw, (record.offset - offset) as usize, record.length)?;
                    let decoded = Page::parse(frame, record.checksum, None)?;
                    let expected = self
                        .manifest
                        .page_positions
                        .min(o.count - page * self.manifest.page_positions)
                        as usize;
                    if decoded.n != expected {
                        return Err("page cell count".into());
                    }
                    let value = Arc::new(Cached::Page(decoded));
                    fill(&value, targets, &mut output)?;
                    self.cache_put(Self::key(o, 2, *page, 0), value)?;
                }
                i = stop;
            }
            begin = end;
        }
        Ok(output)
    }
}
fn fill(value: &Cached, targets: &[(usize, usize)], output: &mut [i64]) -> Result<()> {
    for &(target, local) in targets {
        output[target] = match value {
            Cached::Constant(x) => *x,
            Cached::Page(page) => page.get(local)?,
            _ => return Err("cache kind".into()),
        };
    }
    Ok(())
}
