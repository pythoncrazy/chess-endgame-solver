//! Exact quiet-move fibers. Other pieces stay fixed; captures use a separate path.
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Predicates {
    pub winning: u64,
    pub nonwinning: u64,
    pub legal: u64,
}
fn shift(bits: u64, df: i32, dr: i32, steps: u32) -> u64 {
    let delta = (8 * dr + df) * steps as i32;
    let file_mask = match (df, steps) {
        (1, 1) => 0x7f7f7f7f7f7f7f7f,
        (1, 2) => 0x3f3f3f3f3f3f3f3f,
        (1, 4) => 0x0f0f0f0f0f0f0f0f,
        (-1, 1) => 0xfefefefefefefefe,
        (-1, 2) => 0xfcfcfcfcfcfcfcfc,
        (-1, 4) => 0xf0f0f0f0f0f0f0f0,
        _ => u64::MAX,
    };
    if delta >= 0 {
        (bits & file_mask) << delta
    } else {
        (bits & file_mask) >> (-delta)
    }
}
/// All source squares with at least one target knight destination.
pub fn knight_reach(x: u64) -> u64 {
    ((x & 0x7f7f7f7f7f7f7f7f) << 17)
        | ((x & 0xfefefefefefefefe) << 15)
        | ((x & 0x3f3f3f3f3f3f3f3f) << 10)
        | ((x & 0xfcfcfcfcfcfcfcfc) << 6)
        | ((x & 0x7f7f7f7f7f7f7f7f) >> 15)
        | ((x & 0xfefefefefefefefe) >> 17)
        | ((x & 0x3f3f3f3f3f3f3f3f) >> 6)
        | ((x & 0xfcfcfcfcfcfcfcfc) >> 10)
}
/// Parallel occluded fill with 1/2/4-square stages. Empty is physical occupancy,
/// not the own-king safety mask: unsafe intermediate squares do not block a ray.
pub fn slider_reach(targets: u64, empty: u64, bishop: bool, rook: bool) -> u64 {
    let mut all = 0;
    for (df, dr, enabled) in [
        (1, 0, rook),
        (-1, 0, rook),
        (0, 1, rook),
        (0, -1, rook),
        (1, 1, bishop),
        (-1, 1, bishop),
        (1, -1, bishop),
        (-1, -1, bishop),
    ] {
        if !enabled {
            continue;
        }
        let mut generator = targets;
        let mut propagator = empty;
        generator |= propagator & shift(generator, df, dr, 1);
        propagator &= shift(propagator, df, dr, 1);
        generator |= propagator & shift(generator, df, dr, 2);
        propagator &= shift(propagator, df, dr, 2);
        generator |= propagator & shift(generator, df, dr, 4);
        all |= shift(generator, df, dr, 1) & empty;
    }
    all
}
pub fn stencil(
    w: u64,
    l: u64,
    destination_safe: u64,
    source_valid: u64,
    occupied: u64,
    piece: u8,
) -> Result<Predicates> {
    if w & l != 0 || destination_safe & occupied != 0 {
        return Err("invalid child planes or destination mask".into());
    }
    let reach = |targets| match piece {
        2 => knight_reach(targets),
        3 => slider_reach(targets, !occupied, true, false),
        4 => slider_reach(targets, !occupied, false, true),
        5 => slider_reach(targets, !occupied, true, true),
        _ => 0,
    };
    if !(2..=5).contains(&piece) {
        return Err("unsupported quiet fiber piece".into());
    }
    Ok(Predicates {
        winning: reach(l & destination_safe) & source_valid,
        nonwinning: reach(!w & destination_safe) & source_valid,
        legal: reach(destination_safe) & source_valid,
    })
}
/// INT8 RHS values 1/10/91 encode legal, nonwinning and losing child counts.
/// Knight degree <=8, so base-9 digits cannot carry. No FP rounding is involved.
pub fn encoded_knight(
    w: u64,
    l: u64,
    destination_safe: u64,
    source_valid: u64,
) -> Result<Predicates> {
    if w & l != 0 {
        return Err("overlapping child planes".into());
    }
    let mut result = Predicates {
        winning: 0,
        nonwinning: 0,
        legal: 0,
    };
    for source in 0..64 {
        if source_valid >> source & 1 == 0 {
            continue;
        }
        let children = knight_reach(1 << source) & destination_safe;
        let z = children.count_ones()
            + 9 * (children & !w).count_ones()
            + 81 * (children & l).count_ones();
        if !z.is_multiple_of(9) {
            result.legal |= 1 << source;
        }
        if !(z / 9).is_multiple_of(9) {
            result.nonwinning |= 1 << source;
        }
        if z / 81 != 0 {
            result.winning |= 1 << source;
        }
    }
    Ok(result)
}

/// Division-free 0/16/17 encoding. Move existence is a separate static mask.
pub fn encoded_knight16(
    w: u64,
    l: u64,
    destination_safe: u64,
    source_valid: u64,
) -> Result<Predicates> {
    if w & l != 0 {
        return Err("overlapping child planes".into());
    }
    let mut result = Predicates {
        winning: 0,
        nonwinning: 0,
        legal: 0,
    };
    for source in 0..64 {
        if source_valid >> source & 1 == 0 {
            continue;
        }
        let children = knight_reach(1 << source) & destination_safe;
        let z = 16 * (children & !w).count_ones() + (children & l).count_ones();
        if z & 15 != 0 {
            result.winning |= 1 << source;
        }
        if z != 0 {
            result.nonwinning |= 1 << source;
        }
        if children != 0 {
            result.legal |= 1 << source;
        }
    }
    Ok(result)
}

/// FP8 E4M3-compatible 0/8/9 inputs. Eight losses carry into the high digit;
/// that case is uniquely z=72, because no knight has nine destinations.
pub fn encoded_knight8(
    w: u64,
    l: u64,
    destination_safe: u64,
    source_valid: u64,
) -> Result<Predicates> {
    if w & l != 0 {
        return Err("overlapping child planes".into());
    }
    let mut result = Predicates {
        winning: 0,
        nonwinning: 0,
        legal: 0,
    };
    for source in 0..64 {
        if source_valid >> source & 1 == 0 {
            continue;
        }
        let children = knight_reach(1 << source) & destination_safe;
        let z = 8 * (children & !w).count_ones() + (children & l).count_ones();
        if z & 7 != 0 || z == 72 {
            result.winning |= 1 << source;
        }
        if z != 0 {
            result.nonwinning |= 1 << source;
        }
        if children != 0 {
            result.legal |= 1 << source;
        }
    }
    Ok(result)
}
