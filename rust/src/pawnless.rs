//! Pawnless retrograde solver reference: unclocked WDL with proof ranks.
//!
//! Layout: the skeleton holds every piece except the black king, and
//! `skeleton[0]` is the white king. A word fixes the side to move and every
//! skeleton square. Its 64 bits are the black-king square. No move captures a
//! king. Thus a capture dependency reads an aligned word of a lower material
//! when a skeleton piece was captured. It reads a single bit of one when the
//! black king captured a white piece.
//!
//! Rounds are synchronous: round `r` reads only the sets of round `r - 1`. A
//! position committed in round `r` has proof rank `r`. A win has a legal child
//! loss of rank `r - 1`, or a capture into a lower-material loss. Every legal
//! child of a loss is a win of smaller rank, or a lower-material win.
//! Checkmates have rank 0. Lower-material children count as rank 0. Thus ranks
//! equal distance to mate only for materials without capture dependencies.
use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::thread;

/// At most five skeleton pieces: six-piece materials, with both kings in the count.
pub const MAX_SKELETON: usize = 5;
/// Rank of a position that is not (yet) a proven win or loss.
pub const UNRESOLVED: u16 = u16::MAX;
/// Version 2 stores one record per word: `[win | illegal, loss | illegal]` as
/// little-endian u64. If both bits are set, the position is illegal. In ranked
/// tables, the 64 u16 proof ranks of the word follow. One contiguous read
/// decides any position.
const FILE_MAGIC: &[u8; 8] = b"TBPAWNL2";
/// Version 1 (three separate planes, then ranks). Read-only.
const FILE_MAGIC_V1: &[u8; 8] = b"TBPAWNL1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
}

const ORTHOGONAL: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const DIAGONAL: [(i8, i8); 4] = [(1, 1), (-1, 1), (1, -1), (-1, -1)];
const ALL_DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-1, -1),
];
const KNIGHT_JUMPS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];

impl Role {
    pub fn letter(self) -> char {
        match self {
            Role::King => 'K',
            Role::Queen => 'Q',
            Role::Rook => 'R',
            Role::Bishop => 'B',
            Role::Knight => 'N',
        }
    }
    fn from_letter(letter: char) -> Option<Role> {
        Some(match letter {
            'K' => Role::King,
            'Q' => Role::Queen,
            'R' => Role::Rook,
            'B' => Role::Bishop,
            'N' => Role::Knight,
            _ => return None,
        })
    }
    /// Movement vectors, and a flag that shows if the piece slides along them.
    pub fn movement(self) -> (&'static [(i8, i8)], bool) {
        match self {
            Role::King => (&ALL_DIRECTIONS, false),
            Role::Queen => (&ALL_DIRECTIONS, true),
            Role::Rook => (&ORTHOGONAL, true),
            Role::Bishop => (&DIAGONAL, true),
            Role::Knight => (&KNIGHT_JUMPS, false),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Piece {
    pub white: bool,
    pub role: Role,
}

/// A pawnless material with a fixed skeleton order.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Material {
    pub skeleton: Vec<Piece>,
}

impl Material {
    /// Parse a name such as `KQvKR`. Both sides start with a king. Pieces keep
    /// their written order (white non-kings, then black non-kings).
    pub fn parse(name: &str) -> Result<Material> {
        let (white, black) = name
            .split_once('v')
            .ok_or_else(|| format!("material {name:?} lacks 'v'"))?;
        let side = |text: &str, white: bool| -> Result<Vec<Piece>> {
            let mut chars = text.chars();
            if chars.next() != Some('K') {
                return Err(format!("material {name:?}: each side starts with K"));
            }
            chars
                .map(|letter| match Role::from_letter(letter) {
                    Some(Role::King) | None => Err(format!(
                        "material {name:?}: unsupported piece {letter:?} (pawnless, one king)"
                    )),
                    Some(role) => Ok(Piece { white, role }),
                })
                .collect()
        };
        let mut skeleton = vec![Piece {
            white: true,
            role: Role::King,
        }];
        skeleton.extend(side(white, true)?);
        skeleton.extend(side(black, false)?);
        if skeleton.len() > MAX_SKELETON {
            return Err(format!("material {name:?} has more than six pieces"));
        }
        Ok(Material { skeleton })
    }

    pub fn name(&self) -> String {
        let side = |white: bool| -> String {
            let rest: String = self.skeleton[1..]
                .iter()
                .filter(|p| p.white == white)
                .map(|p| p.role.letter())
                .collect();
            format!("K{rest}")
        };
        format!("{}v{}", side(true), side(false))
    }

    /// Number of skeleton coordinates (all pieces except the black king).
    pub fn skeleton_len(&self) -> usize {
        self.skeleton.len()
    }

    /// Words per table: two turns times 64 squares per skeleton piece.
    pub fn words(&self) -> usize {
        2 * 64usize.pow(self.skeleton.len() as u32)
    }

    /// The material after skeleton piece `index` is captured.
    pub fn without(&self, index: usize) -> Result<Material> {
        if index == 0 || index >= self.skeleton.len() {
            return Err(format!("skeleton piece {index} cannot be captured"));
        }
        let mut skeleton = self.skeleton.clone();
        skeleton.remove(index);
        Ok(Material { skeleton })
    }

    /// Word index: turn (0 white, 1 black) is the most significant digit,
    /// followed by the skeleton squares in order.
    pub fn encode(&self, black_to_move: bool, squares: &[u8]) -> usize {
        debug_assert_eq!(squares.len(), self.skeleton.len());
        squares
            .iter()
            .fold(usize::from(black_to_move), |index, &square| {
                index * 64 + usize::from(square)
            })
    }

    pub fn decode(&self, mut word: usize) -> (bool, [u8; MAX_SKELETON]) {
        let mut squares = [0u8; MAX_SKELETON];
        for slot in squares[..self.skeleton.len()].iter_mut().rev() {
            *slot = (word % 64) as u8;
            word /= 64;
        }
        (word == 1, squares)
    }
}

/// Precomputed leaper attacks and between-square masks.
pub struct Geometry {
    king: [u64; 64],
    knight: [u64; 64],
    between: Vec<u64>,
}

fn step(square: u8, (df, dr): (i8, i8)) -> Option<u8> {
    let file = (square % 8) as i8 + df;
    let rank = (square / 8) as i8 + dr;
    ((0..8).contains(&file) && (0..8).contains(&rank)).then(|| (rank * 8 + file) as u8)
}

impl Default for Geometry {
    fn default() -> Self {
        Self::new()
    }
}

impl Geometry {
    pub fn new() -> Geometry {
        let leaper = |jumps: &[(i8, i8)], square: u8| {
            jumps
                .iter()
                .filter_map(|&jump| step(square, jump))
                .fold(0u64, |mask, target| mask | 1 << target)
        };
        let mut between = vec![0u64; 64 * 64];
        for from in 0..64u8 {
            for &direction in &ALL_DIRECTIONS {
                let mut path = 0u64;
                let mut current = from;
                while let Some(next) = step(current, direction) {
                    between[usize::from(from) * 64 + usize::from(next)] = path;
                    path |= 1 << next;
                    current = next;
                }
            }
        }
        Geometry {
            king: std::array::from_fn(|s| leaper(&ALL_DIRECTIONS, s as u8)),
            knight: std::array::from_fn(|s| leaper(&KNIGHT_JUMPS, s as u8)),
            between,
        }
    }

    pub fn king(&self, square: u8) -> u64 {
        self.king[usize::from(square)]
    }

    /// Squares strictly between two squares on a common line, else zero.
    pub fn between(&self, from: u8, to: u8) -> u64 {
        self.between[usize::from(from) * 64 + usize::from(to)]
    }

    /// Attacked squares, with the first occupied square on each ray.
    pub fn attacks(&self, role: Role, square: u8, occupied: u64) -> u64 {
        match role {
            Role::King => self.king[usize::from(square)],
            Role::Knight => self.knight[usize::from(square)],
            _ => {
                let (directions, _) = role.movement();
                let mut mask = 0;
                for &direction in directions {
                    let mut current = square;
                    while let Some(next) = step(current, direction) {
                        mask |= 1 << next;
                        if occupied >> next & 1 != 0 {
                            break;
                        }
                        current = next;
                    }
                }
                mask
            }
        }
    }
}

/// Bit-parallel king neighbourhood of every set square.
pub fn king_reach(x: u64) -> u64 {
    let not_a = 0xfefe_fefe_fefe_fefe;
    let not_h = 0x7f7f_7f7f_7f7f_7f7f;
    let sideways = ((x & not_h) << 1) | ((x & not_a) >> 1);
    let row = x | sideways;
    sideways | (row << 8) | (row >> 8)
}

/// A solved material in the working layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub material: Material,
    pub valid: Vec<u64>,
    pub win: Vec<u64>,
    pub loss: Vec<u64>,
    /// Proof rank per position (`word * 64 + black king square`). Empty when
    /// the table was solved in place without proof ranks.
    pub rank: Vec<u16>,
    /// Number of synchronous rounds after the checkmate seed.
    pub rounds: u32,
}

/// Static masks of one word: legal positions and positions in check.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Statics {
    pub valid: u64,
    pub check: u64,
}

/// Move-derived masks of one word for the current round.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moves {
    /// Some legal child is a proven loss.
    pub win: u64,
    /// Some legal child is not a proven win.
    pub nonwin: u64,
    /// Some legal move exists.
    pub any: u64,
}

impl Moves {
    fn add(&mut self, mask: u64, win: u64, loss: u64, valid: u64) {
        self.win |= loss & mask;
        self.nonwin |= !win & valid & mask;
        self.any |= valid & mask;
    }
    fn add_reach(&mut self, reach: impl Fn(u64) -> u64, win: u64, loss: u64, valid: u64) {
        self.win |= reach(loss);
        self.nonwin |= reach(!win & valid);
        self.any |= reach(valid);
    }
}

/// Lower-material tables indexed by the captured skeleton piece.
pub type Lower<'a> = [Option<&'a Table>];

fn occupancy(material: &Material, squares: &[u8]) -> Option<u64> {
    let mut occupied = 0u64;
    for &square in &squares[..material.skeleton.len()] {
        if occupied >> square & 1 != 0 {
            return None;
        }
        occupied |= 1 << square;
    }
    Some(occupied)
}

pub fn statics(geometry: &Geometry, material: &Material, word: usize) -> Statics {
    let (black_to_move, squares) = material.decode(word);
    let Some(occupied) = occupancy(material, &squares) else {
        return Statics::default();
    };
    let white_king = squares[0];
    let base = !occupied & !geometry.king(white_king);
    let mut white_attacks = 0u64;
    // Black-king squares where no black piece attacks the white king.
    let mut white_king_safe = u64::MAX;
    for (piece, &square) in material.skeleton.iter().zip(&squares) {
        if piece.white {
            white_attacks |= geometry.attacks(piece.role, square, occupied);
        } else if geometry.attacks(piece.role, square, occupied) >> white_king & 1 != 0 {
            // Only the black king can block, and only on a square between the two pieces.
            white_king_safe &= geometry.between(square, white_king);
        }
    }
    if black_to_move {
        let valid = base & white_king_safe;
        Statics {
            valid,
            check: valid & white_attacks,
        }
    } else {
        let valid = base & !white_attacks;
        Statics {
            valid,
            check: valid & !white_king_safe,
        }
    }
}

/// Planes that a sweep reads: the sets of the material and its lower tables.
pub struct Planes<'a> {
    pub material: &'a Material,
    pub valid: &'a [u64],
    pub win: &'a [u64],
    pub loss: &'a [u64],
    pub lower: &'a Lower<'a>,
}

pub fn moves(geometry: &Geometry, planes: &Planes, word: usize) -> Result<Moves> {
    let material = planes.material;
    let (black_to_move, squares) = material.decode(word);
    let mut out = Moves::default();
    let Some(occupied) = occupancy(material, &squares) else {
        return Ok(out);
    };
    let len = material.skeleton.len();
    for (index, piece) in material.skeleton.iter().enumerate() {
        if piece.white == black_to_move {
            continue;
        }
        let from = squares[index];
        let mut targets = geometry.attacks(piece.role, from, occupied);
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            // White: own king. Black: possible only from an illegal parent.
            // In a valid parent, the black king blocks that ray.
            if to == squares[0] {
                continue;
            }
            let captured = (1..len).find(|&k| squares[k] == to);
            if captured.is_some_and(|k| material.skeleton[k].white == piece.white) {
                continue;
            }
            // The black king on the path blocks the move. On the target, it is illegal.
            let mask = !geometry.between(from, to) & !(1u64 << to);
            let mut child = squares;
            child[index] = to;
            match captured {
                None => {
                    let c = material.encode(!black_to_move, &child[..len]);
                    out.add(mask, planes.win[c], planes.loss[c], planes.valid[c]);
                }
                Some(k) => {
                    let table = lower_table(planes.lower, k)?;
                    let rest = without(&child[..len], k);
                    let c = table.material.encode(!black_to_move, &rest);
                    out.add(mask, table.win[c], table.loss[c], table.valid[c]);
                }
            }
        }
    }
    if black_to_move {
        let c = material.encode(false, &squares[..len]);
        out.add_reach(king_reach, planes.win[c], planes.loss[c], planes.valid[c]);
        for k in 1..len {
            if !material.skeleton[k].white {
                continue;
            }
            let table = lower_table(planes.lower, k)?;
            let to = squares[k];
            let c = table.material.encode(false, &without(&squares[..len], k));
            let bit = |plane: &[u64]| (plane[c] >> to & 1) * u64::MAX;
            let sources = geometry.king(to);
            out.add(
                sources,
                bit(&table.win),
                bit(&table.loss),
                bit(&table.valid),
            );
        }
    }
    Ok(out)
}

fn without(squares: &[u8], index: usize) -> Vec<u8> {
    let mut rest = squares.to_vec();
    rest.remove(index);
    rest
}

fn lower_table<'a>(lower: &'a Lower, index: usize) -> Result<&'a Table> {
    lower
        .get(index)
        .copied()
        .flatten()
        .ok_or_else(|| format!("missing lower table for captured skeleton piece {index}"))
}

/// Run `f` over word ranges in parallel and collect the per-word results.
fn parallel_map<T: Send + Copy + Default>(
    words: usize,
    threads: usize,
    f: impl Fn(usize) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let mut out = vec![T::default(); words];
    let chunk = words.div_ceil(threads.max(1)).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = out
            .chunks_mut(chunk)
            .enumerate()
            .map(|(part, slice)| {
                let f = &f;
                scope.spawn(move || -> Result<()> {
                    for (offset, slot) in slice.iter_mut().enumerate() {
                        *slot = f(part * chunk + offset)?;
                    }
                    Ok(())
                })
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().map_err(|_| "solver thread panicked".to_string())?)
    })?;
    Ok(out)
}

/// Solve a material on the CPU. `lower[k]` must hold the solved material
/// without skeleton piece `k` for every non-king `k`.
pub fn solve(material: &Material, lower: &Lower, threads: usize) -> Result<Table> {
    let geometry = Geometry::new();
    let words = material.words();
    let stat = parallel_map(words, threads, |w| Ok(statics(&geometry, material, w)))?;
    let valid: Vec<u64> = stat.iter().map(|s| s.valid).collect();
    let mut win = vec![0u64; words];
    let mut loss = vec![0u64; words];
    let seed = parallel_map(words, threads, |w| {
        let planes = Planes {
            material,
            valid: &valid,
            win: &win,
            loss: &loss,
            lower,
        };
        moves(&geometry, &planes, w)
    })?;
    let mut rank = vec![UNRESOLVED; words * 64];
    for (w, (s, m)) in stat.iter().zip(&seed).enumerate() {
        loss[w] = s.valid & !m.any & s.check;
        set_rank(&mut rank, w, loss[w], 0);
    }
    let mut rounds = 0u32;
    loop {
        let round = rounds + 1;
        let next = parallel_map(words, threads, |w| {
            let planes = Planes {
                material,
                valid: &valid,
                win: &win,
                loss: &loss,
                lower,
            };
            let m = moves(&geometry, &planes, w)?;
            Ok((
                win[w] | (m.win & valid[w]),
                loss[w] | (valid[w] & m.any & !m.nonwin),
            ))
        })?;
        let mut changed = false;
        for (w, &(new_win, new_loss)) in next.iter().enumerate() {
            let fresh = (new_win & !win[w]) | (new_loss & !loss[w]);
            if fresh != 0 {
                changed = true;
                set_rank(&mut rank, w, fresh, round);
            }
            win[w] = new_win;
            loss[w] = new_loss;
        }
        if !changed {
            break;
        }
        rounds = round;
        if rounds >= u32::from(UNRESOLVED) {
            return Err("proof rank exceeds the 16-bit representation".into());
        }
    }
    if win.iter().zip(&loss).any(|(w, l)| w & l != 0) {
        return Err("win and loss masks overlap".into());
    }
    Ok(Table {
        material: material.clone(),
        valid,
        win,
        loss,
        rank,
        rounds,
    })
}

fn set_rank(rank: &mut [u16], word: usize, mut bits: u64, value: u32) {
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        rank[word * 64 + bit] = value as u16;
    }
}

/// Header of the table file. The SHA-256 covers every byte after the header.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileHeader {
    pub format: String,
    pub material: String,
    pub layout: String,
    pub words: usize,
    pub rounds: u32,
    pub planes: Vec<String>,
    pub payload_sha256: String,
    /// Program and source revision that produced the table. Integrity (the
    /// digest) does not prove correctness, so restarts reuse only tables from
    /// the same generator.
    #[serde(default)]
    pub generator: String,
    /// The method for `payload_sha256`. Empty means SHA-256 of the payload.
    /// `sha256-tree-16MiB` means SHA-256 over the SHA-256 digests of 16 MiB
    /// chunks, then the payload length. The tree scheme is parallel to compute and check.
    #[serde(default)]
    pub digest_scheme: String,
}

const DIGEST_CHUNK: usize = 16 << 20;
pub const DIGEST_TREE: &str = "sha256-tree-16MiB";

/// Payload digest under `scheme`, computed in parallel for the tree scheme.
pub fn payload_digest(payload: &[u8], scheme: &str) -> Result<String> {
    match scheme {
        "" => Ok(format!("{:x}", Sha256::digest(payload))),
        DIGEST_TREE => {
            let chunks: Vec<&[u8]> = payload.chunks(DIGEST_CHUNK).collect();
            let digests = parallel_map(chunks.len(), 16, |i| {
                Ok(<[u8; 32]>::from(Sha256::digest(chunks[i])))
            })?;
            let mut root = Sha256::new();
            for d in &digests {
                root.update(d);
            }
            root.update((payload.len() as u64).to_le_bytes());
            Ok(format!("{:x}", root.finalize()))
        }
        other => Err(format!("unknown digest scheme {other:?}")),
    }
}

impl Table {
    /// Bytes per stored proof rank: none, u8 (255 = unresolved) when every
    /// rank fits, else u16.
    pub fn rank_width(&self) -> usize {
        match (self.rank.is_empty(), self.rounds < 255) {
            (true, _) => 0,
            (false, true) => 1,
            (false, false) => 2,
        }
    }

    /// Bytes per word record in the version 2 file layout.
    pub fn record_len(&self) -> usize {
        16 + 64 * self.rank_width()
    }

    /// Version 2 payload, built in parallel chunks of whole records.
    pub fn payload(&self) -> Vec<u8> {
        let record = self.record_len();
        let mut bytes = vec![0u8; self.valid.len() * record];
        let threads = thread::available_parallelism().map_or(4, |n| n.get().min(16));
        let chunk_words = self.valid.len().div_ceil(threads).max(1);
        thread::scope(|scope| {
            for (part, out) in bytes.chunks_mut(chunk_words * record).enumerate() {
                scope.spawn(move || {
                    for (i, rec) in out.chunks_exact_mut(record).enumerate() {
                        let w = part * chunk_words + i;
                        let illegal = !self.valid[w];
                        rec[..8].copy_from_slice(&(self.win[w] | illegal).to_le_bytes());
                        rec[8..16].copy_from_slice(&(self.loss[w] | illegal).to_le_bytes());
                        let ranks = &self.rank.get(w * 64..w * 64 + 64).unwrap_or(&[]);
                        match record {
                            80 => {
                                for (b, &value) in ranks.iter().enumerate() {
                                    rec[16 + b] = value.min(255) as u8;
                                }
                            }
                            144 => {
                                for (b, value) in ranks.iter().enumerate() {
                                    rec[16 + 2 * b..18 + 2 * b]
                                        .copy_from_slice(&value.to_le_bytes());
                                }
                            }
                            _ => {}
                        }
                    }
                });
            }
        });
        bytes
    }

    pub fn header(&self, payload: &[u8], generator: &str) -> Result<FileHeader> {
        let mut planes = vec![
            "win|illegal:u64".to_string(),
            "loss|illegal:u64".to_string(),
        ];
        match self.rank_width() {
            1 => planes.push("rank:u8x64".to_string()),
            2 => planes.push("rank:u16x64".to_string()),
            _ => {}
        }
        Ok(FileHeader {
            format: "tablebase-pawnless-v2".into(),
            material: self.material.name(),
            layout: "record per word (turn, skeleton squares; white king first); \
                     bits = black king square; win and loss both set = illegal"
                .into(),
            words: self.valid.len(),
            rounds: self.rounds,
            planes,
            payload_sha256: payload_digest(payload, DIGEST_TREE)?,
            generator: generator.to_string(),
            digest_scheme: DIGEST_TREE.to_string(),
        })
    }

    pub fn write(&self, mut out: impl Write, generator: &str) -> Result<FileHeader> {
        let payload = self.payload();
        let header = self.header(&payload, generator)?;
        let json = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
        out.write_all(FILE_MAGIC).map_err(|e| e.to_string())?;
        out.write_all(&(json.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&json).map_err(|e| e.to_string())?;
        out.write_all(&payload).map_err(|e| e.to_string())?;
        Ok(header)
    }

    /// Read and authenticate a table file (version 2, or legacy version 1).
    /// Corrupt or truncated data causes an error.
    pub fn read(mut input: impl Read) -> Result<(Table, FileHeader)> {
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic).map_err(|e| e.to_string())?;
        let v2 = &magic == FILE_MAGIC;
        if !v2 && &magic != FILE_MAGIC_V1 {
            return Err("not a pawnless table file".into());
        }
        let mut len = [0u8; 8];
        input.read_exact(&mut len).map_err(|e| e.to_string())?;
        let len = usize::try_from(u64::from_le_bytes(len)).map_err(|e| e.to_string())?;
        if len > 1 << 20 {
            return Err("table header too large".into());
        }
        let mut json = vec![0u8; len];
        input.read_exact(&mut json).map_err(|e| e.to_string())?;
        let header: FileHeader = serde_json::from_slice(&json).map_err(|e| e.to_string())?;
        let material = Material::parse(&header.material)?;
        let words = header.words;
        if material.words() != words {
            return Err("table word count disagrees with its material".into());
        }
        let rank_width = match (v2, header.planes.len()) {
            (true, 2) | (false, 3) => 0,
            (true, 3) if header.planes[2] == "rank:u8x64" => 1,
            (true, 3) | (false, 4) => 2,
            _ => return Err("table header lists an unsupported plane set".into()),
        };
        let ranked = rank_width > 0;
        let mut payload = Vec::new();
        input.read_to_end(&mut payload).map_err(|e| e.to_string())?;
        let expected = words * if v2 { 16 } else { 24 } + words * 64 * rank_width;
        if payload.len() != expected {
            return Err("table payload has the wrong length".into());
        }
        if payload_digest(&payload, &header.digest_scheme)? != header.payload_sha256 {
            return Err("table payload digest mismatch".into());
        }
        let u64_at = |offset: usize| {
            u64::from_le_bytes(payload[offset..offset + 8].try_into().expect("8 bytes"))
        };
        let u16_at = |offset: usize| u16::from_le_bytes([payload[offset], payload[offset + 1]]);
        let mut table = Table {
            material,
            valid: vec![0; words],
            win: vec![0; words],
            loss: vec![0; words],
            rank: if ranked {
                vec![0; words * 64]
            } else {
                Vec::new()
            },
            rounds: header.rounds,
        };
        if v2 {
            let record = 16 + 64 * rank_width;
            for w in 0..words {
                let (marked_win, marked_loss) = (u64_at(w * record), u64_at(w * record + 8));
                let valid = !(marked_win & marked_loss);
                table.valid[w] = valid;
                table.win[w] = marked_win & valid;
                table.loss[w] = marked_loss & valid;
                for b in 0..64 * usize::from(ranked) {
                    table.rank[w * 64 + b] = if rank_width == 1 {
                        match payload[w * record + 16 + b] {
                            255 => UNRESOLVED,
                            r => u16::from(r),
                        }
                    } else {
                        u16_at(w * record + 16 + 2 * b)
                    };
                }
            }
        } else {
            for w in 0..words {
                table.valid[w] = u64_at(8 * w);
                table.win[w] = u64_at(8 * (words + w));
                table.loss[w] = u64_at(8 * (2 * words + w));
            }
            if ranked {
                for (i, r) in table.rank.iter_mut().enumerate() {
                    *r = u16_at(24 * words + 2 * i);
                }
            }
        }
        if table.win.iter().zip(&table.loss).any(|(w, l)| w & l != 0) {
            return Err("table marks a position both won and lost".into());
        }
        Ok((table, header))
    }
}

/// Solved tables by material name, shared between dependent materials.
pub type Solved = std::collections::BTreeMap<String, std::sync::Arc<Table>>;

/// Solve `material` and, first, every material reachable by captures. The
/// `solve_one` callback receives a material and its complete lower tables.
pub fn solve_tree(
    material: &Material,
    solved: &mut Solved,
    solve_one: &mut dyn FnMut(&Material, &Lower) -> Result<Table>,
) -> Result<std::sync::Arc<Table>> {
    if let Some(table) = solved.get(&material.name()) {
        return Ok(table.clone());
    }
    let mut lower = vec![None; material.skeleton_len()];
    for (k, slot) in lower.iter_mut().enumerate().skip(1) {
        *slot = Some(solve_tree(&material.without(k)?, solved, solve_one)?);
    }
    let refs: Vec<Option<&Table>> = lower.iter().map(|t| t.as_deref()).collect();
    let table = std::sync::Arc::new(solve_one(material, &refs)?);
    solved.insert(material.name(), table.clone());
    Ok(table)
}

/// Position counts by turn and value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counts {
    pub legal: u64,
    pub win: u64,
    pub loss: u64,
    pub draw: u64,
}

impl Table {
    /// Counts for white to move and black to move.
    pub fn counts(&self) -> [Counts; 2] {
        let half = self.valid.len() / 2;
        let count = |range: std::ops::Range<usize>| {
            let mut c = Counts::default();
            for w in range {
                c.legal += u64::from(self.valid[w].count_ones());
                c.win += u64::from(self.win[w].count_ones());
                c.loss += u64::from(self.loss[w].count_ones());
            }
            c.draw = c.legal - c.win - c.loss;
            c
        };
        [count(0..half), count(half..2 * half)]
    }

    /// Unclocked WDL of one position: Some(2), Some(0), Some(-2), or None if illegal.
    pub fn wdl(&self, word: usize, black_king: u8) -> Option<i8> {
        let bit = |plane: &[u64]| plane[word] >> black_king & 1 != 0;
        if !bit(&self.valid) {
            None
        } else if bit(&self.win) {
            Some(2)
        } else if bit(&self.loss) {
            Some(-2)
        } else {
            Some(0)
        }
    }
}

/// One of the eight board symmetries: optional transpose (a1-h8 diagonal),
/// then optional file mirror, then optional rank mirror.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Symmetry {
    pub transpose: bool,
    pub mirror_files: bool,
    pub mirror_ranks: bool,
}

impl Symmetry {
    pub fn all() -> impl Iterator<Item = Symmetry> {
        (0..8u8).map(|b| Symmetry {
            transpose: b & 1 != 0,
            mirror_files: b & 2 != 0,
            mirror_ranks: b & 4 != 0,
        })
    }

    pub fn square(self, mut s: u8) -> u8 {
        if self.transpose {
            s = ((s & 7) << 3) | (s >> 3);
        }
        if self.mirror_files {
            s ^= 7;
        }
        if self.mirror_ranks {
            s ^= 56;
        }
        s
    }

    pub fn bitboard(self, mut x: u64) -> u64 {
        if self.transpose {
            let t = 0x0f0f_0f0f_0000_0000 & (x ^ (x << 28));
            x ^= t ^ (t >> 28);
            let t = 0x3333_0000_3333_0000 & (x ^ (x << 14));
            x ^= t ^ (t >> 14);
            let t = 0x5500_5500_5500_5500 & (x ^ (x << 7));
            x ^= t ^ (t >> 7);
        }
        if self.mirror_files {
            x = ((x >> 1) & 0x5555_5555_5555_5555) | ((x & 0x5555_5555_5555_5555) << 1);
            x = ((x >> 2) & 0x3333_3333_3333_3333) | ((x & 0x3333_3333_3333_3333) << 2);
            x = ((x >> 4) & 0x0f0f_0f0f_0f0f_0f0f) | ((x & 0x0f0f_0f0f_0f0f_0f0f) << 4);
        }
        if self.mirror_ranks {
            x = x.swap_bytes();
        }
        x
    }

    /// Map a bitboard from the image domain back to the source domain.
    /// Bit `bk` of the result equals bit `self.square(bk)` of `y`.
    pub fn inverse_bitboard(self, mut y: u64) -> u64 {
        if self.mirror_ranks {
            y = y.swap_bytes();
        }
        if self.mirror_files {
            y = ((y >> 1) & 0x5555_5555_5555_5555) | ((y & 0x5555_5555_5555_5555) << 1);
            y = ((y >> 2) & 0x3333_3333_3333_3333) | ((y & 0x3333_3333_3333_3333) << 2);
            y = ((y >> 4) & 0x0f0f_0f0f_0f0f_0f0f) | ((y & 0x0f0f_0f0f_0f0f_0f0f) << 4);
        }
        if self.transpose {
            let t = 0x0f0f_0f0f_0000_0000 & (y ^ (y << 28));
            y ^= t ^ (t >> 28);
            let t = 0x3333_0000_3333_0000 & (y ^ (y << 14));
            y ^= t ^ (t >> 14);
            let t = 0x5500_5500_5500_5500 & (y ^ (y << 7));
            y ^= t ^ (t >> 7);
        }
        y
    }
}

/// White-king squares of the a1-d1-d4 triangle. Every position has a
/// symmetric image with the white king there.
pub fn canonical_white_king(square: u8) -> bool {
    let (file, rank) = (square % 8, square / 8);
    file <= 3 && rank <= file
}

/// Sort the squares of identical pieces, in skeleton order.
fn sort_identical(material: &Material, squares: &mut [u8]) {
    let skeleton = &material.skeleton;
    for i in 1..skeleton.len() {
        for j in i + 1..skeleton.len() {
            if skeleton[i] == skeleton[j] && squares[j] < squares[i] {
                squares.swap(i, j);
            }
        }
    }
}

/// Words that differ from a symmetric image, or from the same position with
/// identical pieces in a different order. The check compares the valid, win
/// and loss planes, and the proof ranks of legal positions. Pawnless positions
/// without castling rights are invariant under all eight symmetries, so a
/// correct table has none.
pub fn symmetry_violations(table: &Table, threads: usize) -> Result<u64> {
    let material = &table.material;
    let len = material.skeleton_len();
    let ranked = !table.rank.is_empty();
    let violations = parallel_map(table.valid.len(), threads, |w| {
        let (black, squares) = material.decode(w);
        for sym in Symmetry::all() {
            let mut image: Vec<u8> = squares[..len].iter().map(|&s| sym.square(s)).collect();
            sort_identical(material, &mut image);
            let v = material.encode(black, &image);
            if v == w {
                continue;
            }
            for plane in [&table.valid, &table.win, &table.loss] {
                if sym.bitboard(plane[w]) != plane[v] {
                    return Ok(1);
                }
            }
            if ranked {
                let mut legal = table.valid[w];
                while legal != 0 {
                    let bk = legal.trailing_zeros() as u8;
                    legal &= legal - 1;
                    let image = v * 64 + usize::from(sym.square(bk));
                    if table.rank[w * 64 + usize::from(bk)] != table.rank[image] {
                        return Ok(1);
                    }
                }
            }
        }
        Ok(0u64)
    })?;
    Ok(violations.iter().sum())
}
