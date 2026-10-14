//! Padded pawnless working index. Collisions/checks are separate legality masks.
//! The index quotients only the king pairs. Stabilizer duplicates keep quiet axes regular.
use crate::Result;

pub const KING_PAIRS: u64 = 462;
pub const STRIDE: u64 = 64 * 64 * 64 * 64;
pub const SLOTS: u64 = 2 * KING_PAIRS * STRIDE;

pub fn transform(square: u8, symmetry: u8) -> Result<u8> {
    if square >= 64 || symmetry >= 8 {
        return Err("square/symmetry out of bounds".into());
    }
    let (mut file, mut rank) = (square % 8, square / 8);
    if symmetry & 4 != 0 {
        std::mem::swap(&mut file, &mut rank);
    }
    if symmetry & 1 != 0 {
        file = 7 - file;
    }
    if symmetry & 2 != 0 {
        rank = 7 - rank;
    }
    Ok(rank * 8 + file)
}

#[derive(Clone, Copy)]
pub struct KingMap {
    pub pair: u16,
    pub symmetry: u8,
}
pub struct Dense6 {
    pairs: Vec<[u8; 2]>,
    pub king_map: [Option<KingMap>; 4096],
    pub square_map: [[u8; 64]; 8],
}
impl Dense6 {
    pub fn new() -> Result<Self> {
        let mut square_map = [[0; 64]; 8];
        for (symmetry, map) in square_map.iter_mut().enumerate() {
            for (square, target) in map.iter_mut().enumerate() {
                *target = transform(square as u8, symmetry as u8)?;
            }
        }
        let mut normalized = [None; 4096];
        let mut canonical = std::collections::BTreeSet::new();
        for white in 0u8..64 {
            for black in 0u8..64 {
                if (white % 8).abs_diff(black % 8) <= 1 && (white / 8).abs_diff(black / 8) <= 1 {
                    continue;
                }
                // Tie-break on symmetry makes canonical pairs use identity.
                let (pair, symmetry) = (0u8..8)
                    .map(|s| {
                        (
                            [
                                square_map[s as usize][white as usize],
                                square_map[s as usize][black as usize],
                            ],
                            s,
                        )
                    })
                    .min()
                    .ok_or("no king transform")?;
                normalized[white as usize * 64 + black as usize] = Some((pair, symmetry));
                canonical.insert(pair);
            }
        }
        let pairs: Vec<_> = canonical.into_iter().collect();
        if pairs.len() != KING_PAIRS as usize {
            return Err("king orbit count".into());
        }
        let mut king_map = [None; 4096];
        for (index, item) in normalized.into_iter().enumerate() {
            if let Some((pair, symmetry)) = item {
                king_map[index] = Some(KingMap {
                    pair: pairs.binary_search(&pair).map_err(|_| "king pair absent")? as u16,
                    symmetry,
                });
            }
        }
        Ok(Self {
            pairs,
            king_map,
            square_map,
        })
    }
    pub fn rank(&self, squares: [u8; 6], black_turn: bool) -> Result<u64> {
        if squares.iter().any(|&s| s >= 64) {
            return Err("square out of bounds".into());
        }
        let map = self.king_map[squares[0] as usize * 64 + squares[1] as usize]
            .ok_or("overlapping or adjacent kings")?;
        let mut id = u64::from(black_turn) * KING_PAIRS + u64::from(map.pair);
        for square in squares.into_iter().skip(2) {
            id = id * 64 + u64::from(self.square_map[map.symmetry as usize][square as usize]);
        }
        Ok(id)
    }
    pub fn unrank(&self, mut id: u64) -> Result<([u8; 6], bool)> {
        if id >= SLOTS {
            return Err("dense ID out of bounds".into());
        }
        let mut squares = [0; 6];
        for square in squares[2..].iter_mut().rev() {
            *square = (id % 64) as u8;
            id /= 64;
        }
        let pair = self.pairs[(id % KING_PAIRS) as usize];
        squares[..2].copy_from_slice(&pair);
        Ok((squares, id >= KING_PAIRS))
    }
    /// Coordinate transition only. The caller must give and check a legal quiet move.
    /// A canonical transition of a piece other than a king changes only one base-64 digit.
    pub fn relocate(&self, id: u64, axis: usize, destination: u8) -> Result<u64> {
        if id >= SLOTS || axis >= 6 || destination >= 64 {
            return Err("coordinate transition out of bounds".into());
        }
        if axis >= 2 {
            let stride = 1u64 << ((5 - axis) * 6);
            let old = (id / stride) & 63;
            let child = id - old * stride + u64::from(destination) * stride;
            return Ok(if id >= SLOTS / 2 {
                child - SLOTS / 2
            } else {
                child + SLOTS / 2
            });
        }
        let (mut squares, black_turn) = self.unrank(id)?;
        squares[axis] = destination;
        self.rank(squares, !black_turn)
    }
}
