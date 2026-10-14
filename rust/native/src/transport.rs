//! Read-only range transport and counters. No object uploads or credentials.
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tablebase_core::{Result, archive::RangeStore};
pub struct Metered {
    inner: Arc<dyn RangeStore>,
    pub reads: AtomicU64,
    pub bytes: AtomicU64,
}
impl Metered {
    pub fn new(inner: Arc<dyn RangeStore>) -> Self {
        Self {
            inner,
            reads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }
}
impl RangeStore for Metered {
    fn read(&self, key: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let bytes = self.inner.read(key, offset, length)?;
        self.bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }
}
#[cfg(feature = "http")]
pub struct HttpRanges {
    base: reqwest::Url,
    client: reqwest::blocking::Client,
}
#[cfg(feature = "http")]
impl HttpRanges {
    pub fn new(base: &str) -> Result<Self> {
        let mut base = reqwest::Url::parse(base).map_err(|e| e.to_string())?;
        if !["http", "https"].contains(&base.scheme())
            || base.query().is_some()
            || base.fragment().is_some()
            || !base.username().is_empty()
            || base.password().is_some()
        {
            return Err("plain HTTP(S) range-service base URL required".into());
        }
        if !base.path().ends_with('/') {
            base.set_path(&format!("{}/", base.path()));
        }
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self { base, client })
    }
}
#[cfg(feature = "http")]
impl RangeStore for HttpRanges {
    fn read(&self, key: &str, offset: u64, length: usize) -> Result<Vec<u8>> {
        use std::io::Read;
        if key.len() != 70
            || !key.ends_with(".tbobj")
            || !key[..64]
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || length == 0
            || length > 1 << 20
            || offset
                .checked_add(length as u64)
                .is_none_or(|end| end > (1u64 << 53) - 1)
        {
            return Err("HTTP immutable range bounds".into());
        }
        let url = self
            .base
            .join(&format!("range/{key}/{offset}/{length}"))
            .map_err(|e| e.to_string())?;
        let response = self.client.get(url).send().map_err(|e| e.to_string())?;
        if response.status() != reqwest::StatusCode::OK
            || response
                .content_length()
                .is_some_and(|n| n != length as u64)
        {
            return Err("invalid range-service status/length".into());
        }
        let mut bytes = Vec::with_capacity(length);
        response
            .take(length as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() != length {
            return Err("short/oversized HTTP range resource".into());
        }
        Ok(bytes)
    }
}
