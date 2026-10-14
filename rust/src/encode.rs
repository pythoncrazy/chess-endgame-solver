//! Bounded per-page exact encoder. Compression is an offline build concern.
use crate::{
    Result,
    codec::{Page, hash128},
};
use std::collections::BTreeMap;

fn pack(values: &[u64], width: u8, planes: bool) -> Vec<u8> {
    let mut bytes = vec![
        0u8;
        if planes {
            width as usize * values.len().div_ceil(8)
        } else {
            (values.len() * width as usize).div_ceil(8)
        }
    ];
    for (i, &value) in values.iter().enumerate() {
        if !planes {
            let bit = i * width as usize;
            let shifted = value << (bit % 8);
            for b in 0..(bit % 8 + width as usize).div_ceil(8) {
                bytes[bit / 8 + b] |= (shifted >> (8 * b)) as u8;
            }
            continue;
        }
        for b in 0..width as usize {
            let bit = b * values.len().div_ceil(8) * 8 + i;
            bytes[bit / 8] |= (((value >> b) & 1) as u8) << (bit % 8);
        }
    }
    bytes
}
fn width(x: u64) -> u8 {
    (64 - x.leading_zeros()) as u8
}
pub fn encode(values: &[i64], zstd_level: i32) -> Result<Vec<u8>> {
    if values.len() > 1 << 20 {
        return Err("page exceeds maximum".into());
    }
    let wide = values.iter().any(|&v| i32::try_from(v).is_err());
    let mut decoded = Vec::with_capacity(values.len() * if wide { 8 } else { 4 });
    for &value in values {
        if wide {
            decoded.extend_from_slice(&value.to_le_bytes());
        } else {
            decoded.extend_from_slice(&(value as i32).to_le_bytes());
        }
    }
    let check = hash128(&decoded);
    let mut best: Option<Vec<u8>> = None;
    let mut candidate = |mode: u8, bits: u8, body: Vec<u8>| -> Result<()> {
        let compressed = zstd::bulk::compress(&body, zstd_level).map_err(|e| e.to_string())?;
        let (compression, payload) = if compressed.len() < body.len() {
            (1, compressed.as_slice())
        } else {
            (0, body.as_slice())
        };
        let mut frame = Vec::with_capacity(32 + payload.len());
        frame.extend_from_slice(if wide { b"TBC2" } else { b"TBC1" });
        frame.extend_from_slice(&[mode, compression, bits, 0]);
        frame.extend_from_slice(&(values.len() as u32).to_le_bytes());
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        frame.extend_from_slice(&check);
        frame.extend_from_slice(payload);
        if best.as_ref().is_none_or(|old| frame.len() < old.len()) {
            best = Some(frame);
        }
        Ok(())
    };
    if values.is_empty() || values.iter().all(|&v| v == values[0]) {
        let v = values.first().copied().unwrap_or(0);
        candidate(
            0,
            0,
            if wide {
                v.to_le_bytes().to_vec()
            } else {
                (v as i32).to_le_bytes().to_vec()
            },
        )?;
    } else if wide {
        candidate(7, 64, decoded.clone())?;
        let mut planes = vec![0; decoded.len()];
        for (i, bytes) in decoded.as_chunks::<8>().0.iter().enumerate() {
            for b in 0..8 {
                planes[b * values.len() + i] = bytes[b];
            }
        }
        candidate(9, 64, planes)?;
        for lag in [1, 8, 64].into_iter().filter(|&lag| lag <= values.len()) {
            let mut body = (lag as u32).to_le_bytes().to_vec();
            for (i, &v) in values.iter().enumerate() {
                body.extend_from_slice(
                    &((v as u64) ^ if i >= lag { values[i - lag] as u64 } else { 0 }).to_le_bytes(),
                );
            }
            candidate(8, 64, body)?;
        }
    } else {
        let base = *values.iter().min().unwrap();
        let ranks: Vec<_> = values.iter().map(|&x| (x - base) as u64).collect();
        let bits = width(*ranks.iter().max().unwrap());
        let mut body = (base as i32).to_le_bytes().to_vec();
        body.extend(pack(&ranks, bits, false));
        candidate(1, bits, body)?;
        let mut palette = BTreeMap::new();
        for &v in values {
            palette.insert(v, 0u64);
            if palette.len() > 256 {
                break;
            }
        }
        if palette.len() <= 256 {
            for (rank, value) in palette.values_mut().enumerate() {
                *value = rank as u64;
            }
            let ranks: Vec<_> = values.iter().map(|v| palette[v]).collect();
            let bits = width(palette.len() as u64 - 1);
            let mut prefix = (palette.len() as u16).to_le_bytes().to_vec();
            for &v in palette.keys() {
                prefix.extend_from_slice(&(v as i32).to_le_bytes());
            }
            for planes in [false, true] {
                let mut body = prefix.clone();
                body.extend(pack(&ranks, bits, planes));
                candidate(if planes { 6 } else { 2 }, bits, body)?;
            }
        }
        for lag in [1, 8, 64].into_iter().filter(|&lag| lag <= values.len()) {
            let ranks: Vec<_> = values
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    let r = v - if i >= lag { values[i - lag] } else { 0 };
                    ((r << 1) ^ (r >> 63)) as u64
                })
                .collect();
            let bits = width(*ranks.iter().max().unwrap());
            let mut body = (lag as u32).to_le_bytes().to_vec();
            body.extend(pack(&ranks, bits, true));
            candidate(3, bits, body)?;
        }
    }
    let frame = best.ok_or("encoder produced no candidate")?;
    let page = Page::parse(&frame, None, None)?;
    for (i, &value) in values.iter().enumerate() {
        if page.get(i)? != value {
            return Err("encoder round-trip".into());
        }
    }
    Ok(frame)
}
