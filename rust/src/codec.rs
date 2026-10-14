//! Python-compatible TBC1/TBC2 frames. No floats or lossy integer conversion.
use crate::Result;
use blake2::{Blake2b, Digest, digest::consts::U16};
use std::io::{Cursor, Read};

pub fn hash128(bytes: &[u8]) -> [u8; 16] {
    Blake2b::<U16>::digest(bytes).into()
}
pub(crate) fn slice(raw: &[u8], offset: usize, length: usize) -> Result<&[u8]> {
    raw.get(offset..offset.checked_add(length).ok_or("range overflow")?)
        .ok_or_else(|| "truncated payload".into())
}
pub(crate) fn u32le(raw: &[u8], at: usize) -> Result<u32> {
    Ok(u32::from_le_bytes(slice(raw, at, 4)?.try_into().unwrap()))
}
pub(crate) fn u64le(raw: &[u8], at: usize) -> Result<u64> {
    Ok(u64::from_le_bytes(slice(raw, at, 8)?.try_into().unwrap()))
}
fn i32le(raw: &[u8], at: usize) -> Result<i64> {
    Ok(u32le(raw, at)? as i32 as i64)
}
fn unzigzag(x: u64) -> i64 {
    ((x >> 1) as i64) ^ -((x & 1) as i64)
}

#[derive(Debug)]
pub struct Page {
    pub n: usize,
    mode: u8,
    width: usize,
    wide: bool,
    body: Vec<u8>,
    expected: [u8; 16],
    reconstructed: Option<Vec<i64>>,
}
impl Page {
    /// With an authenticated v3 index checksum, retain direct-access packed bodies.
    /// Without one, reconstruct and verify every cell before returning the page.
    pub fn parse(
        frame: &[u8],
        authenticated: Option<[u8; 16]>,
        prediction: Option<&[i64]>,
    ) -> Result<Self> {
        if let Some(check) = authenticated
            && hash128(frame) != check
        {
            return Err("encoded checksum mismatch".into());
        }
        if frame.len() < 32 {
            return Err("truncated frame".into());
        }
        let wide = match &frame[..4] {
            b"TBC1" => false,
            b"TBC2" => true,
            _ => return Err("frame magic".into()),
        };
        let (mode, compression, width) = (frame[4], frame[5], frame[6] as usize);
        let n = u32le(frame, 8)? as usize;
        let size = u32le(frame, 12)? as usize;
        if frame[7] != 0
            || compression > 2
            || n > 1 << 20
            || frame.len() != 32 + size
            || (wide && ![0, 7, 8, 9].contains(&mode))
            || (!wide && mode > 6)
            || (n == 0 && (mode != 0 || wide))
            || ([0, 4].contains(&mode) && width != 0)
            || (wide && mode != 0 && width != 64)
            || (!wide && width > 33)
        {
            return Err("invalid frame header".into());
        }
        let payload = &frame[32..];
        let limit = 16 * n + 4096;
        let mut body = Vec::new();
        match compression {
            0 => body.extend_from_slice(payload),
            1 => {
                let mut decoder = zstd::stream::read::Decoder::with_buffer(Cursor::new(payload))
                    .map_err(|e| e.to_string())?
                    .single_frame();
                decoder.window_log_max(26).map_err(|e| e.to_string())?;
                let mut bounded = decoder.take((limit + 1) as u64);
                bounded.read_to_end(&mut body).map_err(|e| e.to_string())?;
                if body.len() > limit {
                    return Err("decompression limit".into());
                }
                let cursor = bounded.into_inner().finish();
                if cursor.position() != payload.len() as u64 {
                    return Err("trailing zstd data".into());
                }
            }
            2 => {
                let stream = xz2::stream::Stream::new_stream_decoder(64 << 20, 0)
                    .map_err(|e| e.to_string())?;
                let decoder = xz2::bufread::XzDecoder::new_stream(Cursor::new(payload), stream);
                let mut bounded = decoder.take((limit + 1) as u64);
                bounded.read_to_end(&mut body).map_err(|e| e.to_string())?;
                if body.len() > limit {
                    return Err("decompression limit".into());
                }
                let decoder = bounded.into_inner();
                if decoder.total_in() != payload.len() as u64 {
                    return Err("trailing LZMA data".into());
                }
            }
            _ => unreachable!(),
        }
        if body.len() > limit {
            return Err("payload limit".into());
        }
        let mut page = Self {
            n,
            mode,
            width,
            wide,
            body,
            expected: frame[16..32].try_into().unwrap(),
            reconstructed: None,
        };
        page.validate_layout()?;
        if authenticated.is_none() || ![0, 1, 2, 6, 7, 9].contains(&mode) {
            let values = page.reconstruct(prediction)?;
            let mut hasher = Blake2b::<U16>::new();
            for &value in &values {
                if wide {
                    hasher.update(value.to_le_bytes());
                } else {
                    hasher.update(
                        i32::try_from(value)
                            .map_err(|_| "decoded i32 overflow")?
                            .to_le_bytes(),
                    );
                }
            }
            if <[u8; 16]>::from(hasher.finalize()) != page.expected {
                return Err("decoded checksum mismatch".into());
            }
            page.reconstructed = Some(values);
        }
        Ok(page)
    }
    fn validate_bits(&self, raw: &[u8], planes: bool) -> Result<()> {
        let expected = if planes {
            self.width * self.n.div_ceil(8)
        } else {
            (self.n * self.width).div_ceil(8)
        };
        if raw.len() != expected {
            return Err("packed length".into());
        }
        if self.width == 0 {
            return Ok(());
        }
        if planes && !self.n.is_multiple_of(8) {
            let stride = self.n.div_ceil(8);
            for b in 0..self.width {
                if raw[(b + 1) * stride - 1] >> (self.n % 8) != 0 {
                    return Err("plane padding".into());
                }
            }
        } else if !planes
            && !(self.n * self.width).is_multiple_of(8)
            && raw.last().copied().unwrap_or(0) >> ((self.n * self.width) % 8) != 0
        {
            return Err("packed padding".into());
        }
        Ok(())
    }
    fn palette_end(&self) -> Result<usize> {
        let count = u16::from_le_bytes(slice(&self.body, 0, 2)?.try_into().unwrap()) as usize;
        if !(1..=256).contains(&count) {
            return Err("palette count".into());
        }
        slice(&self.body, 2, 4 * count)?;
        Ok(2 + 4 * count)
    }
    fn validate_layout(&self) -> Result<()> {
        match self.mode {
            0 => {
                if self.body.len() != if self.wide { 8 } else { 4 } {
                    return Err("constant size".into());
                }
                if self.n == 0 && i32le(&self.body, 0)? != 0 {
                    return Err("empty constant".into());
                }
            }
            1 => {
                slice(&self.body, 0, 4)?;
                self.validate_bits(&self.body[4..], false)?;
            }
            2 | 6 => {
                let end = self.palette_end()?;
                self.validate_bits(&self.body[end..], self.mode == 6)?;
            }
            3 => {
                let lag = u32le(&self.body, 0)? as usize;
                if lag == 0 || lag > self.n {
                    return Err("lag".into());
                }
                self.validate_bits(&self.body[4..], true)?;
            }
            4 => {
                slice(&self.body, 0, 20)?;
            }
            5 => {
                slice(&self.body, 0, 16)?;
                self.validate_bits(&self.body[16..], true)?;
            }
            7 | 9 => {
                if self.body.len() != 8 * self.n {
                    return Err("wide size".into());
                }
            }
            8 => {
                if self.body.len() != 4 + 8 * self.n {
                    return Err("wide residual size".into());
                }
                let lag = u32le(&self.body, 0)? as usize;
                if lag == 0 || lag > self.n {
                    return Err("wide lag".into());
                }
            }
            _ => return Err("unknown mode".into()),
        }
        Ok(())
    }
    fn rank(&self, raw: &[u8], i: usize, planes: bool) -> u64 {
        if planes {
            let stride = self.n.div_ceil(8);
            (0..self.width).fold(0, |value, b| {
                value | (((raw[b * stride + i / 8] >> (i % 8)) & 1) as u64) << b
            })
        } else {
            let bit = i * self.width;
            let bytes = (bit % 8 + self.width).div_ceil(8);
            let value = (0..bytes).fold(0, |value, b| value | (raw[bit / 8 + b] as u64) << (8 * b));
            (value >> (bit % 8)) & ((1u64 << self.width) - 1)
        }
    }
    fn direct(&self, i: usize) -> Result<i64> {
        match self.mode {
            0 => {
                if self.wide {
                    Ok(u64le(&self.body, 0)? as i64)
                } else {
                    i32le(&self.body, 0)
                }
            }
            1 => Ok(i32le(&self.body, 0)? + self.rank(&self.body[4..], i, false) as i64),
            2 | 6 => {
                let end = self.palette_end()?;
                let rank = self.rank(&self.body[end..], i, self.mode == 6) as usize;
                if rank >= (end - 2) / 4 {
                    return Err("palette rank".into());
                }
                i32le(&self.body, 2 + rank * 4)
            }
            7 => Ok(u64le(&self.body, 8 * i)? as i64),
            9 => {
                Ok((0..8).fold(0, |v, b| v | (self.body[b * self.n + i] as u64) << (8 * b)) as i64)
            }
            _ => Err("reconstruction required".into()),
        }
    }
    fn reconstruct(&self, prediction: Option<&[i64]>) -> Result<Vec<i64>> {
        if [0, 1, 2, 6, 7, 9].contains(&self.mode) {
            return (0..self.n).map(|i| self.direct(i)).collect();
        }
        let mut values = vec![0i64; self.n];
        if self.mode == 3 || self.mode == 8 {
            let lag = u32le(&self.body, 0)? as usize;
            for i in 0..self.n {
                if self.mode == 8 {
                    values[i] = (u64le(&self.body, 4 + 8 * i)?
                        ^ if i >= lag { values[i - lag] as u64 } else { 0 })
                        as i64;
                } else {
                    let residual = unzigzag(self.rank(&self.body[4..], i, true));
                    values[i] = residual
                        .checked_add(if i >= lag { values[i - lag] } else { 0 })
                        .ok_or("residual overflow")?;
                }
            }
            return Ok(values);
        }
        let pred = prediction.ok_or("predictor required")?;
        if pred.len() != self.n {
            return Err("predictor count".into());
        }
        let mut bytes = Vec::with_capacity(self.n * 4);
        for &p in pred {
            bytes.extend_from_slice(
                &i32::try_from(p)
                    .map_err(|_| "predictor i32 overflow")?
                    .to_le_bytes(),
            );
        }
        if hash128(&bytes).as_slice() != &self.body[..16] {
            return Err("predictor hash".into());
        }
        values.copy_from_slice(pred);
        if self.mode == 5 {
            for (i, value) in values.iter_mut().enumerate() {
                *value = value
                    .checked_add(unzigzag(self.rank(&self.body[16..], i, true)))
                    .ok_or("predictor overflow")?;
            }
        } else {
            let count = u32le(&self.body, 16)? as usize;
            if count > self.n {
                return Err("exception count".into());
            }
            let mut offset = 20;
            let mut next = 0usize;
            for _ in 0..count {
                next = next
                    .checked_add(read_varint(&self.body, &mut offset)? as usize)
                    .ok_or("exception overflow")?;
                if next >= self.n {
                    return Err("exception index".into());
                }
                values[next] = values[next]
                    .checked_add(unzigzag(read_varint(&self.body, &mut offset)?))
                    .ok_or("exception value overflow")?;
                next += 1;
            }
            if offset != self.body.len() {
                return Err("trailing exceptions".into());
            }
        }
        Ok(values)
    }
    pub fn get(&self, i: usize) -> Result<i64> {
        if i >= self.n {
            return Err("cell out of range".into());
        }
        if let Some(values) = &self.reconstructed {
            return Ok(values[i]);
        }
        let value = self.direct(i)?;
        if !self.wide && i32::try_from(value).is_err() {
            return Err("decoded i32 overflow".into());
        }
        Ok(value)
    }
    pub fn retained_bytes(&self) -> usize {
        self.body.len()
            + self.reconstructed.as_ref().map_or(0, |v| v.len() * 8)
            + std::mem::size_of::<Self>()
    }
}
fn read_varint(raw: &[u8], offset: &mut usize) -> Result<u64> {
    let mut value = 0;
    for shift in (0..42).step_by(7) {
        let byte = *raw.get(*offset).ok_or("truncated varint")?;
        *offset += 1;
        value |= ((byte & 127) as u64) << shift;
        if byte < 128 {
            return Ok(value);
        }
    }
    Err("oversized varint".into())
}
