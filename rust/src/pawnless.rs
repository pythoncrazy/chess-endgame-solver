//! Pawnless retrograde solver reference: unclocked WDL with proof ranks.
//!
//! Layout: the skeleton holds every piece except the black king, and
//! `skeleton[0]` is the white king. A word fixes the side to move and every
//! skeleton square; its 64 bits are the black-king square. Kings are never
//! captured, so a capture dependency reads either an aligned word of a lower
//! material (a skeleton piece was captured) or a single bit of one (the black
//! king captured a white piece).
//!
//! Rounds are synchronous: round `r` reads the sets of round `r - 1` only. A
//! position committed in round `r` has proof rank `r`: a win has a legal child
//! loss of rank `r - 1` (or a capture into a lower-material loss), and every
//! legal child of a loss is a win of smaller rank (or a lower-material win).
//! Checkmates have rank 0. Lower-material children count as rank 0, so ranks
//! equal distance to mate only for materials without capture dependencies.
use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::thread;

/// At most five skeleton pieces: six-piece materials including both kings.
pub const MAX_SKELETON: usize = 5;
/// Rank of a position that is not (yet) a proven win or loss.
pub const UNRESOLVED: u16 = u16::MAX;
const FILE_MAGIC: &[u8; 8] = b"TBPAWNL1";

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
    /// Movement vectors and whether the piece slides along them.
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
    /// Parse a name such as `KQvKR`. Both sides start with a king; pieces keep
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

    /// Attacked squares, including the first occupied square on each ray.
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
    /// Proof rank per position (`word * 64 + black king square`).
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
    // Black-king squares that leave the white king unattacked by black pieces.
    let mut white_king_safe = u64::MAX;
    for (piece, &square) in material.skeleton.iter().zip(&squares) {
        if piece.white {
            white_attacks |= geometry.attacks(piece.role, square, occupied);
        } else if geometry.attacks(piece.role, square, occupied) >> white_king & 1 != 0 {
            // Only the black king can block, and only by standing between.
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

/// Planes read by a sweep: the material's own sets and its lower tables.
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
            // White: own king. Black: only possible from an illegal parent,
            // because a valid parent has the black king blocking that ray.
            if to == squares[0] {
                continue;
            }
            let captured = (1..len).find(|&k| squares[k] == to);
            if captured.is_some_and(|k| material.skeleton[k].white == piece.white) {
                continue;
            }
            // The black king on the path blocks; on the target it is illegal.
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
}

impl Table {
    pub fn payload(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.valid.len() * 24 + self.rank.len() * 2);
        for plane in [&self.valid, &self.win, &self.loss] {
            for word in plane {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        for value in &self.rank {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    pub fn header(&self, payload: &[u8]) -> FileHeader {
        FileHeader {
            format: "tablebase-pawnless-v1".into(),
            material: self.material.name(),
            layout: "turn,skeleton squares (white king first); bits = black king square".into(),
            words: self.valid.len(),
            rounds: self.rounds,
            planes: ["valid:u64", "win:u64", "loss:u64", "rank:u16"]
                .map(String::from)
                .to_vec(),
            payload_sha256: format!("{:x}", Sha256::digest(payload)),
        }
    }

    pub fn write(&self, mut out: impl Write) -> Result<FileHeader> {
        let payload = self.payload();
        let header = self.header(&payload);
        let json = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
        out.write_all(FILE_MAGIC).map_err(|e| e.to_string())?;
        out.write_all(&(json.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        out.write_all(&json).map_err(|e| e.to_string())?;
        out.write_all(&payload).map_err(|e| e.to_string())?;
        Ok(header)
    }

    /// Read and authenticate a table file; corrupt or truncated data fails.
    pub fn read(mut input: impl Read) -> Result<(Table, FileHeader)> {
        let mut magic = [0u8; 8];
        input.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != FILE_MAGIC {
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
        if material.words() != header.words {
            return Err("table word count disagrees with its material".into());
        }
        let mut payload = Vec::new();
        input.read_to_end(&mut payload).map_err(|e| e.to_string())?;
        if payload.len() != header.words * (24 + 128) {
            return Err("table payload has the wrong length".into());
        }
        if format!("{:x}", Sha256::digest(&payload)) != header.payload_sha256 {
            return Err("table payload digest mismatch".into());
        }
        let plane = |index: usize| -> Vec<u64> {
            payload[index * header.words * 8..(index + 1) * header.words * 8]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&b| u64::from_le_bytes(b))
                .collect()
        };
        let rank = payload[header.words * 24..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect();
        let table = Table {
            material,
            valid: plane(0),
            win: plane(1),
            loss: plane(2),
            rank,
            rounds: header.rounds,
        };
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
