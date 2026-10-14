//! Context mixer codec (`cm`) for `.tbz` sections.
//!
//! The codec codes each stored value as a short sequence of binary decisions.
//! The contexts of a decision come from the values before it in the block.
//! Other contexts come from the chess geometry of the position.
//! A fixed model gives each context a probability. A fixed mixer adds these
//! probabilities in the logistic domain. A binary arithmetic coder then codes
//! the decision.
//!
//! The model is in the section. Thus each block is independent of the other
//! blocks. The encoder codes all blocks in parallel, and a probe decodes one
//! small block. Specification: `TBZ_FORMAT.md`.
use crate::Result;
use crate::pawnless::{Geometry, Piece};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

/// Value of a position that the block does not code.
pub const UNKNOWN: u8 = 4;
/// Rank of a position that the block does not code.
pub const NO_RANK: u16 = u16::MAX;
const MAGIC: &[u8; 4] = b"CM1\0";

const SQUASH_POINTS: [i32; 33] = [
    1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348,
    3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
];

/// 4096 / (1 + e^(-x / 256)), with linear steps between 33 points.
const fn squash(x: i32) -> i32 {
    if x > 2047 {
        return 4095;
    }
    if x < -2047 {
        return 1;
    }
    let w = x & 127;
    let i = ((x >> 7) + 16) as usize;
    (SQUASH_POINTS[i] * (128 - w) + SQUASH_POINTS[i + 1] * w + 64) >> 7
}

/// Inverse of `squash` for each 12-bit probability.
const STRETCH: [i16; 4096] = {
    let mut t = [0i16; 4096];
    let mut next = 0usize;
    let mut x = -2047;
    while x <= 2047 {
        let v = squash(x) as usize;
        while next <= v {
            t[next] = x as i16;
            next += 1;
        }
        x += 1;
    }
    while next < 4096 {
        t[next] = 2047;
        next += 1;
    }
    t
};

/// A fixed model: a stretched probability for each context of each model,
/// and the mixer weights for each weight set.
pub struct Model {
    /// Stretched probability of a 1 bit, or 0 if the context has no data.
    tables: Vec<Vec<i16>>,
    /// Weights in 16.16 fixed point, `sets` rows of one weight for each model.
    weights: Vec<i32>,
}

impl Model {
    /// Probability of a 1 bit in 12 bits.
    #[inline]
    fn p(&self, contexts: &[u32], set: usize) -> u32 {
        let n = self.tables.len();
        let w = &self.weights[set * n..(set + 1) * n];
        let mut dot = 0i64;
        for ((t, &c), &wi) in self.tables.iter().zip(contexts).zip(w) {
            dot += i64::from(wi) * i64::from(t[c as usize]);
        }
        squash((dot >> 16).clamp(-2047, 2047) as i32) as u32
    }

    /// Read a model blob. The sizes and the number of sets must agree.
    pub fn parse(blob: &[u8], sizes: &[usize], sets: usize) -> Result<Model> {
        let raw = zstd::stream::decode_all(blob).map_err(|e| e.to_string())?;
        let mut at = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let s = raw.get(at..at + n).ok_or("cm model is too short")?;
            at += n;
            Ok(s)
        };
        let u32_at = |b: &[u8]| u32::from_le_bytes(b.try_into().expect("4 bytes")) as usize;
        if take(4)? != MAGIC {
            return Err("cm model has the wrong magic".into());
        }
        if u32_at(take(4)?) != sizes.len() || u32_at(take(4)?) != sets {
            return Err("cm model has the wrong shape".into());
        }
        let mut tables = Vec::with_capacity(sizes.len());
        for &size in sizes {
            if u32_at(take(4)?) != size {
                return Err("cm model has the wrong table size".into());
            }
            let present = take(size.div_ceil(8))?.to_vec();
            let count = present
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum::<usize>();
            let probs = take(2 * count)?;
            let mut table = vec![0i16; size];
            let mut k = 0;
            for (i, slot) in table.iter_mut().enumerate() {
                if present[i / 8] >> (i % 8) & 1 == 1 {
                    let p = u16::from_le_bytes([probs[2 * k], probs[2 * k + 1]]);
                    if !(1..4096).contains(&p) {
                        return Err("cm model probability out of range".into());
                    }
                    *slot = STRETCH[usize::from(p)];
                    k += 1;
                }
            }
            tables.push(table);
        }
        let weights = take(4 * sets * sizes.len())?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b))
            .collect();
        if at != raw.len() {
            return Err("cm model has extra bytes".into());
        }
        Ok(Model { tables, weights })
    }
}

/// Binary arithmetic coder with 32-bit bounds, as in lpaq.
struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    fn new() -> Encoder {
        Encoder {
            x1: 0,
            x2: u32::MAX,
            out: Vec::new(),
        }
    }

    #[inline]
    fn code(&mut self, bit: bool, p: u32) {
        let mid = self.x1 + ((self.x2 - self.x1) >> 12) * p;
        if bit {
            self.x2 = mid;
        } else {
            self.x1 = mid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
    }

    /// The decoder reads 255 after the last byte.
    fn finish(mut self) -> Vec<u8> {
        self.out.push((self.x1 >> 24) as u8);
        self.out
    }
}

struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    data: &'a [u8],
    at: usize,
}

impl Decoder<'_> {
    fn new(data: &[u8]) -> Decoder<'_> {
        let mut d = Decoder {
            x1: 0,
            x2: u32::MAX,
            x: 0,
            data,
            at: 0,
        };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.byte();
        }
        d
    }

    #[inline]
    fn byte(&mut self) -> u32 {
        let b = self.data.get(self.at).copied().unwrap_or(255);
        self.at += 1;
        u32::from(b)
    }

    #[inline]
    fn code(&mut self, p: u32) -> bool {
        let mid = self.x1 + ((self.x2 - self.x1) >> 12) * p;
        let bit = self.x <= mid;
        if bit {
            self.x2 = mid;
        } else {
            self.x1 = mid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.byte();
        }
        bit
    }
}

/// A user of binary decisions. The walk gives the contexts, the weight set and
/// the bit of the encoder. The result is the bit to use: the encoder gives back
/// its bit, and the decoder gives the decoded bit.
pub trait Decide {
    fn decide(&mut self, contexts: &[u32], set: usize, bit: bool) -> bool;
}

struct Encode<'a> {
    model: &'a Model,
    coder: Encoder,
}

impl Decide for Encode<'_> {
    #[inline]
    fn decide(&mut self, contexts: &[u32], set: usize, bit: bool) -> bool {
        self.coder.code(bit, self.model.p(contexts, set));
        bit
    }
}

struct Decode<'a, 'b> {
    model: &'a Model,
    coder: Decoder<'b>,
}

impl Decide for Decode<'_, '_> {
    #[inline]
    fn decide(&mut self, contexts: &[u32], set: usize, _: bool) -> bool {
        self.coder.code(self.model.p(contexts, set))
    }
}

/// Bit counts of each context.
struct Count(Vec<Vec<[u32; 2]>>);

impl Decide for Count {
    fn decide(&mut self, contexts: &[u32], _: usize, bit: bool) -> bool {
        for (t, &c) in self.0.iter_mut().zip(contexts) {
            t[c as usize][usize::from(bit)] += 1;
        }
        bit
    }
}

/// Online trainer of the mixer weights with fixed tables.
struct Train<'a> {
    tables: &'a [Vec<i16>],
    weights: Vec<f32>,
    inputs: Vec<f32>,
    rate: f32,
}

impl Decide for Train<'_> {
    fn decide(&mut self, contexts: &[u32], set: usize, bit: bool) -> bool {
        let n = self.tables.len();
        let w = &mut self.weights[set * n..(set + 1) * n];
        let mut dot = 0.0;
        for ((s, t), (&c, &wi)) in self
            .inputs
            .iter_mut()
            .zip(self.tables)
            .zip(contexts.iter().zip(w.iter()))
        {
            *s = f32::from(t[c as usize]) / 256.0;
            dot += wi * *s;
        }
        let p = 1.0 / (1.0 + (-dot).exp());
        let err = (f32::from(u8::from(bit)) - p) * self.rate;
        for (wi, s) in w.iter_mut().zip(&self.inputs) {
            *wi += err * s;
        }
        bit
    }
}

/// The blocks of one section, as the encoder sees them.
pub trait Blocks: Sync {
    /// Number of contexts of each model.
    fn sizes(&self) -> Vec<usize>;
    /// Number of mixer weight sets.
    fn sets(&self) -> usize;
    fn blocks(&self) -> usize;
    /// Send the decisions of block `b` to `d`.
    fn walk<D: Decide>(&self, b: usize, d: &mut D);
}

/// Settings of the encoder. They change the file, but not the format.
#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// A context must have this number of bits or more to get a probability.
    pub prune: u32,
    /// The trainer uses one block in `sample`.
    pub sample: usize,
    pub epochs: usize,
    pub rate: f32,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            prune: 64,
            sample: 8,
            epochs: 2,
            rate: 0.002,
        }
    }
}

/// Memory limit of the bit counters of the encoder.
const COUNT_BYTES: usize = 4 << 30;

/// Number of trainers. Each trainer has a fixed part of the sample, so the
/// weights do not depend on the number of threads.
const TRAINERS: usize = 8;

/// Run `f` on each block with one state for each thread, and give the states.
fn per_thread<S: Send>(
    blocks: usize,
    threads: usize,
    make: impl Fn() -> S + Sync,
    f: impl Fn(usize, &mut S) + Sync,
) -> Vec<S> {
    let next = AtomicUsize::new(0);
    let out = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, blocks.max(1)) {
            scope.spawn(|| {
                let mut state = make();
                loop {
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    if b >= blocks {
                        break;
                    }
                    f(b, &mut state);
                }
                out.lock().expect("states").push(state);
            });
        }
    });
    out.into_inner().expect("states")
}

/// Build the model of a section and code each block.
/// The result is the model blob and one frame for each block.
pub fn encode<B: Blocks>(
    section: &B,
    settings: Settings,
    threads: usize,
) -> Result<(Vec<u8>, Vec<Vec<u8>>)> {
    let sizes = section.sizes();
    let sets = section.sets();
    let blocks = section.blocks();
    let empty = || Count(sizes.iter().map(|&n| vec![[0u32; 2]; n]).collect());
    let per_counter = 8 * sizes.iter().sum::<usize>();
    let counters = threads.min(COUNT_BYTES / per_counter.max(1)).max(1);
    let mut parts = per_thread(blocks, counters, empty, |b, c| section.walk(b, c));
    let mut counts = parts.pop().expect("one counter");
    std::thread::scope(|scope| {
        for (i, t) in counts.0.iter_mut().enumerate() {
            let parts = &parts;
            scope.spawn(move || {
                for part in parts {
                    for (a, b) in t.iter_mut().zip(&part.0[i]) {
                        a[0] += b[0];
                        a[1] += b[1];
                    }
                }
            });
        }
    });
    drop(parts);
    let mut raw = Vec::new();
    raw.extend_from_slice(MAGIC);
    raw.extend_from_slice(&(sizes.len() as u32).to_le_bytes());
    raw.extend_from_slice(&(sets as u32).to_le_bytes());
    let mut tables = Vec::with_capacity(sizes.len());
    for t in &counts.0 {
        let mut present = vec![0u8; t.len().div_ceil(8)];
        let mut probs = Vec::new();
        let table = t
            .iter()
            .enumerate()
            .map(|(i, &[n0, n1])| {
                let n = u64::from(n0) + u64::from(n1);
                if n < u64::from(settings.prune.max(1)) {
                    return 0;
                }
                let (num, den) = ((10 * u64::from(n1) + 4) * 4096, 10 * n + 8);
                let p = ((2 * num + den) / (2 * den)).clamp(1, 4095) as u16;
                present[i / 8] |= 1 << (i % 8);
                probs.extend_from_slice(&p.to_le_bytes());
                STRETCH[usize::from(p)]
            })
            .collect::<Vec<i16>>();
        raw.extend_from_slice(&(t.len() as u32).to_le_bytes());
        raw.extend_from_slice(&present);
        raw.extend_from_slice(&probs);
        tables.push(table);
    }
    drop(counts);
    let n = sizes.len();
    let sample: Vec<usize> = (0..blocks).step_by(settings.sample.max(1)).collect();
    let trained = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..TRAINERS)
            .map(|t| {
                let (tables, sample) = (&tables, &sample);
                scope.spawn(move || {
                    let mut trainer = Train {
                        tables,
                        weights: vec![0.3; sets * n],
                        inputs: vec![0.0; n],
                        rate: settings.rate,
                    };
                    for _ in 0..settings.epochs {
                        for &b in sample.iter().skip(t).step_by(TRAINERS) {
                            section.walk(b, &mut trainer);
                        }
                    }
                    trainer.weights
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("trainer"))
            .collect::<Vec<_>>()
    });
    let weights: Vec<i32> = (0..sets * n)
        .map(|i| {
            let mean = trained.iter().map(|w| f64::from(w[i])).sum::<f64>() / TRAINERS as f64;
            (mean * 65536.0)
                .round()
                .clamp(-(1 << 24) as f64, (1 << 24) as f64) as i32
        })
        .collect();
    for w in &weights {
        raw.extend_from_slice(&w.to_le_bytes());
    }
    let model = Model { tables, weights };
    let frames = std::sync::Mutex::new(vec![Vec::new(); blocks]);
    per_thread(
        blocks,
        threads,
        || (),
        |b, _| {
            let mut e = Encode {
                model: &model,
                coder: Encoder::new(),
            };
            section.walk(b, &mut e);
            frames.lock().expect("frames")[b] = e.coder.finish();
        },
    );
    let blob = zstd::bulk::compress(&raw, 19).map_err(|e| e.to_string())?;
    Ok((blob, frames.into_inner().expect("frames")))
}

/// One word of a block: the stored legal positions and the skeleton squares.
#[derive(Clone, Copy, Debug, Default)]
pub struct Word {
    pub stored: u64,
    pub black: bool,
    pub squares: [u8; 5],
}

fn distance(a: u8, b: u8) -> i32 {
    (i32::from(a % 8) - i32::from(b % 8))
        .abs()
        .max((i32::from(a / 8) - i32::from(b / 8)).abs())
}

pub const GEOMETRY: usize = 16 * 16 * 16 * 3;

/// Attack data of one word, without the black king.
struct Board {
    /// Attacks of each piece when the black king is not on the board.
    attacks: [u64; 5],
    /// For each target piece and side: the squares between each attacker and the target.
    lines: [[[u64; 4]; 2]; 5],
    counts: [[usize; 2]; 5],
    squares: [u8; 5],
}

impl Board {
    fn new(g: &Geometry, pieces: &[Piece], squares: &[u8]) -> Board {
        let occupied = squares.iter().fold(0u64, |o, &s| o | 1 << s);
        let mut b = Board {
            attacks: [0; 5],
            lines: [[[0; 4]; 2]; 5],
            counts: [[0; 2]; 5],
            squares: [0; 5],
        };
        b.squares[..squares.len()].copy_from_slice(squares);
        for (j, p) in pieces.iter().enumerate() {
            b.attacks[j] = g.attacks(p.role, squares[j], occupied);
        }
        for target in 1..pieces.len() {
            for (j, p) in pieces.iter().enumerate() {
                if j != target && b.attacks[j] >> squares[target] & 1 != 0 {
                    let side = usize::from(p.white);
                    let n = b.counts[target][side];
                    b.lines[target][side][n] = g.between(squares[j], squares[target]);
                    b.counts[target][side] += 1;
                }
            }
        }
        b
    }

    /// Geometry context: the white checks, the loose pieces and the king distance.
    /// A loose piece is a piece that the other side can take without a recapture.
    fn geometry(&self, g: &Geometry, pieces: &[Piece], bk: u8) -> usize {
        let bit = 1u64 << bk;
        let near = g.king(bk);
        let attacked = |side: usize, target: usize| {
            (side == 0 && near >> self.squares[target] & 1 != 0)
                || self.lines[target][side][..self.counts[target][side]]
                    .iter()
                    .any(|&l| l & bit == 0)
        };
        let (mut check, mut white_loose, mut black_loose) = (0, 0, 0);
        for (j, p) in pieces.iter().enumerate().skip(1) {
            let mark = 1 << (j - 1);
            if p.white && self.attacks[j] & bit != 0 {
                check |= mark;
            }
            let own = usize::from(p.white);
            if attacked(1 - own, j) && !attacked(own, j) {
                if p.white {
                    white_loose |= mark;
                } else {
                    black_loose |= mark;
                }
            }
        }
        let kings = (distance(self.squares[0], bk) - 2).clamp(0, 2) as usize;
        ((check * 16 + white_loose) * 16 + black_loose) * 3 + kings
    }
}

/// Small geometry context: a check, a loose white piece, a loose black piece
/// and the king distance.
fn small_geometry(geo: usize) -> usize {
    let (kings, black, white, check) = (geo % 3, geo / 3 % 16, geo / 48 % 16, geo / 768);
    ((usize::from(check > 0) * 2 + usize::from(white > 0)) * 2 + usize::from(black > 0)) * 3 + kings
}

/// Value at (file + df, rank + dr) of word cw - dw, if the block has it.
#[inline]
fn near<T: Copy>(v: &[T], none: T, cw: usize, bk: usize, (dw, df, dr): (usize, i32, i32)) -> T {
    let (f, r) = ((bk % 8) as i32 + df, (bk / 8) as i32 + dr);
    if cw < dw || !(0..8).contains(&f) || !(0..8).contains(&r) {
        return none;
    }
    v[(cw - dw) * 64 + (r * 8 + f) as usize]
}

/// Bucket of a run length: 0, 1, 2-3, 4-7, up to 64 and more.
fn run_bucket(run: u32) -> usize {
    (run + 1).ilog2().min(7) as usize
}

pub const WDL_NODES: usize = 6;
pub const WDL_SETS: usize = WDL_NODES * 5;
const WDL_CONTEXTS: [usize; 14] = [
    1,
    25,
    625,
    15625,
    15625,
    78125,
    3125,
    3125,
    200,
    1000,
    200,
    GEOMETRY,
    GEOMETRY * 25,
    GEOMETRY * 25,
];

pub fn wdl_sizes() -> Vec<usize> {
    WDL_CONTEXTS.iter().map(|n| n * WDL_NODES).collect()
}

/// Walk the WDL decisions of a block. `values` has 64 entries for each word.
/// On entry it has the codes of the encoder (or anything for the decoder).
/// On exit it has the coded values, and UNKNOWN where nothing is stored.
pub fn walk_wdl<D: Decide>(
    g: &Geometry,
    pieces: &[Piece],
    words: &[Word],
    values: &mut [u8],
    d: &mut D,
) {
    let mut c = [0u32; 14];
    let unknown = usize::from(UNKNOWN);
    let (mut previous, mut run_prev, mut run_w1, mut run_left) = (unknown, 0u32, 0u32, 0u32);
    for (cw, word) in words.iter().enumerate() {
        let truth: [u8; 64] = values[cw * 64..(cw + 1) * 64].try_into().expect("64");
        values[cw * 64..(cw + 1) * 64].fill(UNKNOWN);
        let turn = usize::from(word.black) * 3;
        let board = Board::new(g, pieces, &word.squares[..pieces.len()]);
        let mut bits = word.stored;
        while bits != 0 {
            let bk = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let geo = board.geometry(g, pieces, bk as u8);
            let at = |o| usize::from(near(values, UNKNOWN, cw, bk, o));
            let (left, up, ul, ur) = (
                at((0, -1, 0)),
                at((0, 0, -1)),
                at((0, -1, -1)),
                at((0, 1, -1)),
            );
            let (w1, w8, w64, w512, w4096) = (
                at((1, 0, 0)),
                at((8, 0, 0)),
                at((64, 0, 0)),
                at((512, 0, 0)),
                at((4096, 0, 0)),
            );
            let (w1l, w1u) = (at((1, -1, 0)), at((1, 0, -1)));
            let square = left * 5 + up;
            let word5 = square * 25 + w1 * 5 + w8;
            let base = [
                0,
                square,
                word5,
                word5 * 25 + ul * 5 + ur,
                word5 * 25 + w64 * 5 + w512,
                (word5 * 25 + ul * 5 + ur) * 5 + w64,
                (((w1 * 5 + w8) * 5 + w64) * 5 + w512) * 5 + w4096,
                (((square * 5 + w1) * 5 + w1l) * 5) + w1u,
                (w1 * 8 + run_bucket(run_w1)) * 5 + left,
                ((square * 8 + run_bucket(run_left)) * 5) + w1,
                (previous * 8 + run_bucket(run_prev)) * 5 + up,
                geo,
                geo * 25 + square,
                geo * 25 + w1 * 5 + w8,
            ];
            let mut node = |n: usize, bit: bool| {
                for (ci, b) in c.iter_mut().zip(base) {
                    *ci = (b * WDL_NODES + n + turn) as u32;
                }
                d.decide(&c, (n + turn) * 5 + left, bit)
            };
            let t = truth[bk];
            let v = if node(0, t == 0) {
                0
            } else if node(1, t == 2) {
                2
            } else if turn != 0 && node(2, t == 3) {
                3
            } else {
                1
            };
            values[cw * 64 + bk] = v;
            let v = usize::from(v);
            run_prev = if v == previous { run_prev + 1 } else { 0 };
            run_w1 = if v == w1 { run_w1 + 1 } else { 0 };
            run_left = if v == left { run_left + 1 } else { 0 };
            previous = v;
        }
    }
}

const RANK_CONTEXTS: [usize; 9] = [8, 2056, 648, 11664, 11664, 9252, 11664, 2 * GEOMETRY, 15552];

/// Decisions of one rank: zero, sign, then the unary length and the low bits
/// for each sign.
pub fn rank_nodes(width: usize) -> usize {
    let k = 8 * width;
    2 + 2 * (k + k * k)
}

pub fn rank_sizes(width: usize) -> Vec<usize> {
    RANK_CONTEXTS
        .iter()
        .map(|n| n * rank_nodes(width))
        .collect()
}

pub fn rank_sets(width: usize) -> usize {
    rank_nodes(width) * 2
}

/// Context value of a neighbour rank relative to the predicted rank.
fn rel(x: u16, pred: u16) -> usize {
    if x == NO_RANK {
        17
    } else {
        ((i32::from(x) - i32::from(pred)).clamp(-8, 8) + 8) as usize
    }
}

/// Walk the rank decisions of a block of white-to-move words.
/// `codes` has the WDL codes of the block. The walk codes the rank of each
/// stored position with code 1 or 2. `ranks` works as `values` in `walk_wdl`.
pub fn walk_rank<D: Decide>(
    g: &Geometry,
    pieces: &[Piece],
    words: &[Word],
    codes: &[u8],
    ranks: &mut [u16],
    width: usize,
    d: &mut D,
) {
    let (k_bits, nodes) = (8 * width, rank_nodes(width));
    let half = k_bits + k_bits * k_bits;
    let mut c = [0u32; 9];
    for (cw, word) in words.iter().enumerate() {
        let truth: [u16; 64] = ranks[cw * 64..(cw + 1) * 64].try_into().expect("64");
        ranks[cw * 64..(cw + 1) * 64].fill(NO_RANK);
        let board = Board::new(g, pieces, &word.squares[..pieces.len()]);
        let mut bits = word.stored;
        while bits != 0 {
            let bk = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let code = codes[cw * 64 + bk];
            if code != 1 && code != 2 {
                continue;
            }
            let win = usize::from(code == 1);
            let geo = board.geometry(g, pieces, bk as u8);
            let at = |o| near(ranks, NO_RANK, cw, bk, o);
            let (left, up, ul, ur) = (
                at((0, -1, 0)),
                at((0, 0, -1)),
                at((0, -1, -1)),
                at((0, 1, -1)),
            );
            let (w1, w8, w64) = (at((1, 0, 0)), at((8, 0, 0)), at((64, 0, 0)));
            let (kind, pred) = [left, up, w1]
                .iter()
                .position(|&x| x != NO_RANK)
                .map_or((3, 0), |i| (i, [left, up, w1][i]));
            let p = usize::from(pred.min(256));
            let base = [
                win * 4 + kind,
                (win * 4 + kind) * 257 + p,
                (win * 18 + rel(up, pred)) * 18 + rel(w1, pred),
                ((win * 18 + rel(up, pred)) * 18 + rel(ul, pred)) * 18 + rel(ur, pred),
                ((win * 18 + rel(w1, pred)) * 18 + rel(w8, pred)) * 18 + rel(w64, pred),
                (win * 257 + p) * 18 + rel(up, pred),
                ((win * 18 + rel(left, pred)) * 18 + rel(w1, pred)) * 18 + rel(w8, pred),
                win * GEOMETRY + geo,
                ((win * 24 + small_geometry(geo)) * 18 + rel(left, pred)) * 18 + rel(up, pred),
            ];
            let mut node = |n: usize, bit: bool| {
                for (ci, b) in c.iter_mut().zip(base) {
                    *ci = (b * nodes + n) as u32;
                }
                d.decide(&c, n * 2 + win, bit)
            };
            let diff = i32::from(truth[bk]) - i32::from(pred);
            let r = if node(0, diff == 0) {
                pred
            } else {
                let neg = node(1, diff < 0);
                let off = 2 + usize::from(neg) * half;
                let mag_true = diff.unsigned_abs().max(1);
                let k_true = mag_true.ilog2() as usize;
                let mut k = 0;
                while k + 1 < k_bits && node(off + k, k < k_true) {
                    k += 1;
                }
                let mut mag = 1u32;
                for i in (0..k).rev() {
                    let bit = node(off + k_bits + k * k_bits + i, mag_true >> i & 1 == 1);
                    mag = mag << 1 | u32::from(bit);
                }
                let r = if neg {
                    i64::from(pred) - i64::from(mag)
                } else {
                    i64::from(pred) + i64::from(mag)
                };
                r.clamp(0, i64::from(NO_RANK - 1)) as u16
            };
            ranks[cw * 64 + bk] = r;
        }
    }
}

/// Decode the WDL codes of a block from its frame.
pub fn decode_wdl(
    model: &Model,
    g: &Geometry,
    pieces: &[Piece],
    words: &[Word],
    frame: &[u8],
) -> Vec<u8> {
    let mut values = vec![UNKNOWN; words.len() * 64];
    let mut d = Decode {
        model,
        coder: Decoder::new(frame),
    };
    walk_wdl(g, pieces, words, &mut values, &mut d);
    values
}

/// Decode the ranks of a block from its frame and the WDL codes of the block.
pub fn decode_rank(
    model: &Model,
    g: &Geometry,
    pieces: &[Piece],
    words: &[Word],
    codes: &[u8],
    width: usize,
    frame: &[u8],
) -> Vec<u16> {
    let mut ranks = vec![NO_RANK; words.len() * 64];
    let mut d = Decode {
        model,
        coder: Decoder::new(frame),
    };
    walk_rank(g, pieces, words, codes, &mut ranks, width, &mut d);
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretch_inverts_squash() {
        for (p, &x) in STRETCH.iter().enumerate().take(4095).skip(1) {
            let x = i32::from(x);
            assert!(squash(x) as usize >= p && (x == -2047 || (squash(x - 1) as usize) < p));
        }
        assert_eq!(squash(0), 2047);
    }

    #[test]
    fn coder_round_trip() {
        let bits: Vec<(bool, u32)> = (0..10_000u32)
            .map(|i| {
                let p = 1 + (i.wrapping_mul(2_654_435_761) >> 20) % 4095;
                ((i.wrapping_mul(40_503) >> 7) % 4096 < p, p)
            })
            .collect();
        let mut e = Encoder::new();
        for &(b, p) in &bits {
            e.code(b, p);
        }
        let out = e.finish();
        let mut d = Decoder::new(&out);
        for &(b, p) in &bits {
            assert_eq!(d.code(p), b);
        }
    }
}
