//! Compare the `.tbz` section sizes with adaptive code lengths.
//! The tool codes the stored WDL and rank values with different contexts.
//!
//! The models skip positions that a reader can find without the file:
//! illegal positions and positions that are not stored. Each model starts
//! again at each block, as a block coder must. The result is the size of an
//! ideal arithmetic coder with that model, without block headers.
use std::{path::PathBuf, time::Instant};
use tablebase_core::{
    packed::{self, BLOCK, Canonical, Layout, PackedTable},
    pawnless::{Piece, Role},
};

const UNKNOWN: u8 = 4;
const NO_RANK: u16 = 256;

/// Adaptive frequency counts per context (Krichevsky-Trofimov estimator).
struct Model {
    symbols: usize,
    counts: Vec<u32>,
    totals: Vec<u32>,
    touched: Vec<usize>,
    bits: f64,
}

impl Model {
    fn new(contexts: usize, symbols: usize) -> Model {
        Model {
            symbols,
            counts: vec![0; contexts * symbols],
            totals: vec![0; contexts],
            touched: Vec::new(),
            bits: 0.0,
        }
    }

    fn reset(&mut self) {
        for c in self.touched.drain(..) {
            self.totals[c] = 0;
            self.counts[c * self.symbols..(c + 1) * self.symbols].fill(0);
        }
    }

    fn code(&mut self, context: usize, symbol: usize) {
        let n = &mut self.counts[context * self.symbols + symbol];
        let total = &mut self.totals[context];
        if *total == 0 {
            self.touched.push(context);
        }
        self.bits +=
            (f64::from(*total) + 0.5 * self.symbols as f64).log2() - (f64::from(*n) + 0.5).log2();
        *n += 1;
        *total += 1;
    }
}

/// Get the WDL code of each stored position, or UNKNOWN.
/// The codes are 0 draw, 1 win, 2 loss and 3 black-to-move win of rank 1.
/// Also get the rank of each white-to-move decisive position.
fn values(layout: &Layout, c: &Canonical) -> (Vec<u8>, Vec<u16>) {
    let words = c.records.len();
    let black_start = words / 2;
    let mut wdl = vec![UNKNOWN; words * 64];
    let mut rank = vec![NO_RANK; words * 64];
    for (cw, &[win, loss]) in c.records.iter().enumerate() {
        let stored = layout.stored_in_word(layout.full_word(cw)) & !(win & loss);
        for bk in 0..64 {
            if stored >> bk & 1 == 0 {
                continue;
            }
            let p = cw * 64 + bk;
            let mut v = (win >> bk & 1) as u8 | (((loss >> bk & 1) as u8) << 1);
            if v == 1 && cw >= black_start && c.rank[p] == 1 {
                v = 3;
            }
            wdl[p] = v;
            if v != 0 && cw < black_start {
                rank[p] = c.rank[p];
            }
        }
    }
    (wdl, rank)
}

fn wdl_bits(wdl: &[u8], words: usize) -> Vec<(&'static str, f64)> {
    let per_block = BLOCK / 16;
    let mut models = [
        ("order0", Model::new(1, 4)),
        ("previous", Model::new(5, 4)),
        ("left+up", Model::new(25, 4)),
        ("left+up+word-1+word-8", Model::new(625, 4)),
        ("left+up+diagonals+word-1+word-8", Model::new(15625, 4)),
        (
            "left+up+word-1+word-8+word-64+word-512",
            Model::new(15625, 4),
        ),
        (
            "left+up+diagonals+word-1+word-8+word-64",
            Model::new(78125, 4),
        ),
    ];
    let mut best = 0.0;
    for start in (0..words).step_by(per_block) {
        let end = (start + per_block).min(words);
        let before: Vec<f64> = models.iter().map(|(_, m)| m.bits).collect();
        for (_, m) in &mut models {
            m.reset();
        }
        let mut previous = UNKNOWN;
        // Value at (file + df, rank + dr) of word cw - dw, if decoded before.
        let at = |cw: usize, bk: i32, dw: usize, df: i32, dr: i32| -> usize {
            let (f, r) = (bk % 8 + df, bk / 8 + dr);
            if cw < start + dw || !(0..8).contains(&f) || !(0..8).contains(&r) {
                return usize::from(UNKNOWN);
            }
            usize::from(wdl[(cw - dw) * 64 + (r * 8 + f) as usize])
        };
        for cw in start..end {
            for bk in 0..64i32 {
                let v = wdl[cw * 64 + bk as usize];
                if v == UNKNOWN {
                    continue;
                }
                let s = usize::from(v);
                let (left, up) = (at(cw, bk, 0, -1, 0), at(cw, bk, 0, 0, -1));
                let (ul, ur) = (at(cw, bk, 0, -1, -1), at(cw, bk, 0, 1, -1));
                let (w1, w8) = (at(cw, bk, 1, 0, 0), at(cw, bk, 8, 0, 0));
                let board = left * 5 + up;
                let word = board * 25 + w1 * 5 + w8;
                models[0].1.code(0, s);
                models[1].1.code(usize::from(previous), s);
                models[2].1.code(board, s);
                models[3].1.code(word, s);
                let (w64, w512) = (at(cw, bk, 64, 0, 0), at(cw, bk, 512, 0, 0));
                models[4].1.code(word * 25 + ul * 5 + ur, s);
                models[5].1.code(word * 25 + w64 * 5 + w512, s);
                models[6].1.code((word * 25 + ul * 5 + ur) * 5 + w64, s);
                previous = v;
            }
        }
        best += models
            .iter()
            .zip(&before)
            .map(|((_, m), b)| m.bits - b)
            .fold(f64::INFINITY, f64::min)
            + 8.0;
    }
    let mut out: Vec<_> = models.into_iter().map(|(n, m)| (n, m.bits)).collect();
    out.push(("best model per block", best));
    out
}

fn bucket(d: u16) -> usize {
    match d {
        0 => 0,
        1 => 1,
        2 => 2,
        3..=4 => 3,
        5..=8 => 4,
        _ => 5,
    }
}

fn rank_bits(wdl: &[u8], rank: &[u16], words: usize) -> Vec<(&'static str, f64)> {
    let per_block = BLOCK / 64;
    let mut models = [
        ("order0 per wdl", Model::new(2, 256)),
        ("wdl+left", Model::new(2 * 257, 256)),
        ("delta to left/up/word-1", Model::new(2 * 4, 256)),
        ("delta, wdl+predictor+|left-up|", Model::new(2 * 4 * 7, 256)),
        ("wdl+left+word-1", Model::new(2 * 257 * 257, 256)),
    ];
    let mut best = 0.0;
    for start in (0..words).step_by(per_block) {
        let end = (start + per_block).min(words);
        let before: Vec<f64> = models.iter().map(|(_, m)| m.bits).collect();
        for (_, m) in &mut models {
            m.reset();
        }
        let at = |cw: usize, bk: usize, dw: usize, dk: usize| -> u16 {
            if cw < start + dw || bk < dk || (dk == 1 && bk.is_multiple_of(8)) {
                return NO_RANK;
            }
            rank[(cw - dw) * 64 + bk - dk]
        };
        for cw in start..end {
            for bk in 0..64 {
                let r = rank[cw * 64 + bk];
                if r == NO_RANK {
                    continue;
                }
                let s = usize::from(r);
                let win = usize::from(wdl[cw * 64 + bk] == 1);
                let (left, up, w1) = (at(cw, bk, 0, 1), at(cw, bk, 0, 8), at(cw, bk, 1, 0));
                let (kind, pred) = [left, up, w1]
                    .iter()
                    .position(|&x| x != NO_RANK)
                    .map_or((3, 0), |i| (i, [left, up, w1][i]));
                let delta = usize::from(r.wrapping_sub(pred) & 255);
                let spread = if left == NO_RANK || up == NO_RANK {
                    6
                } else {
                    bucket(left.abs_diff(up))
                };
                models[0].1.code(win, s);
                models[1].1.code(win * 257 + usize::from(left), s);
                models[2].1.code(win * 4 + kind, delta);
                models[3].1.code((win * 4 + kind) * 7 + spread, delta);
                models[4]
                    .1
                    .code((win * 257 + usize::from(left)) * 257 + usize::from(w1), s);
            }
        }
        best += models
            .iter()
            .zip(&before)
            .map(|((_, m), b)| m.bits - b)
            .fold(f64::INFINITY, f64::min)
            + 8.0;
    }
    let mut out: Vec<_> = models.into_iter().map(|(n, m)| (n, m.bits)).collect();
    out.push(("best model per block", best));
    out
}

/// Binary context mixer, as in PAQ coders.
/// Each model keeps an adaptive bit probability per context.
/// A mixer adds the model outputs in the logistic domain with online weights.
struct Mixer {
    tables: Vec<Vec<(f32, u16)>>,
    weights: Vec<f32>,
    stretched: Vec<f32>,
    refine: Vec<(f32, u16)>,
    limit: u16,
    rate: f32,
    sse: bool,
    bits: f64,
}

fn stretch(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

fn squash(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Mixer {
    fn new(sizes: &[usize], sets: usize, limit: u16, rate: f32) -> Mixer {
        Mixer {
            tables: sizes.iter().map(|&n| vec![(0.5, 0); n]).collect(),
            weights: vec![0.0; sets * sizes.len()],
            stretched: vec![0.0; sizes.len()],
            refine: vec![(0.0, 0); sets * 33],
            limit,
            rate,
            sse: setting("TB_SSE", 1) == 1,
            bits: 0.0,
        }
    }

    fn reset(&mut self) {
        for t in &mut self.tables {
            t.fill((0.5, 0));
        }
        let start = 1.0 / self.tables.len() as f32;
        self.weights.fill(start);
        for (i, r) in self.refine.iter_mut().enumerate() {
            *r = (squash((i % 33) as f32 / 2.0 - 8.0), 0);
        }
    }

    fn code(&mut self, contexts: &[usize], set: usize, bit: bool) {
        let n = self.tables.len();
        let w = &mut self.weights[set * n..(set + 1) * n];
        let mut dot = 0.0;
        for (i, &c) in contexts.iter().enumerate() {
            let s = stretch(self.tables[i][c].0);
            self.stretched[i] = s;
            dot += w[i] * s;
        }
        let mixed = squash(dot);
        let y = f32::from(u8::from(bit));
        let pos = ((dot.clamp(-7.999, 7.999) + 8.0) * 2.0).max(0.0);
        let (lo, frac) = (pos as usize, pos.fract());
        let cell = set * 33 + lo;
        let refined = self.refine[cell].0 * (1.0 - frac) + self.refine[cell + 1].0 * frac;
        let p = if self.sse {
            (mixed + 3.0 * refined) / 4.0
        } else {
            mixed
        }
        .clamp(1.0 / 65536.0, 1.0 - 1.0 / 65536.0);
        self.bits -= f64::from(if bit { p } else { 1.0 - p }).log2();
        for (c, share) in [(cell, 1.0 - frac), (cell + 1, frac)] {
            let (q, k) = &mut self.refine[c];
            *q += (y - *q) * share / (f32::from(*k) + 1.5);
            *k = (*k + 1).min(127);
        }
        let err = (y - mixed) * self.rate;
        for (wi, s) in w.iter_mut().zip(&self.stretched) {
            *wi += err * s;
        }
        for (i, &c) in contexts.iter().enumerate() {
            let (q, k) = &mut self.tables[i][c];
            *q = (*q + (y - *q) / (f32::from(*k) + 1.5)).clamp(1.0 / 4096.0, 1.0 - 1.0 / 4096.0);
            *k = (*k + 1).min(self.limit);
        }
    }
}

/// A consumer of binary decisions with their contexts.
trait Sink {
    fn code(&mut self, contexts: &[usize], set: usize, bit: bool);
    fn block(&mut self) {}
}

impl Sink for Mixer {
    fn code(&mut self, contexts: &[usize], set: usize, bit: bool) {
        Mixer::code(self, contexts, set, bit);
    }
    fn block(&mut self) {
        self.reset();
    }
}

/// A source of binary decisions: one section of a table.
trait Walk: Sync {
    /// Number of contexts of each model.
    fn sizes(&self) -> Vec<usize>;
    /// Number of mixer weight sets.
    fn sets(&self) -> usize;
    /// Send the decisions of the words start..end to the sink.
    fn walk<S: Sink>(&self, start: usize, end: usize, sink: &mut S);
}

/// Walk each block in parallel with one sink for each thread.
fn per_block<W: Walk, S: Sink + Send>(
    walk: &W,
    words: usize,
    per_block: usize,
    blocks: impl Iterator<Item = usize> + Clone + Send,
    make: impl Fn() -> S + Sync,
) -> Vec<S> {
    let threads = 24;
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (make, blocks) = (&make, blocks.clone());
                s.spawn(move || {
                    let mut sink = make();
                    for b in blocks.skip(t).step_by(threads) {
                        sink.block();
                        let start = b * per_block;
                        walk.walk(start, (start + per_block).min(words), &mut sink);
                    }
                    sink
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("block"))
            .collect()
    })
}

/// Code each block with an adaptive mixer that starts again at each block.
fn adaptive<W: Walk>(walk: &W, words: usize, block_words: usize) -> f64 {
    let (limit, rate) = (setting("TB_LIMIT", 255), setting("TB_RATE", 0.005));
    let sizes = walk.sizes();
    per_block(
        walk,
        words,
        block_words,
        0..words.div_ceil(block_words),
        || Mixer::new(&sizes, walk.sets(), limit, rate),
    )
    .iter()
    .map(|m| m.bits)
    .sum()
}

/// Bit counts of each context.
struct Counts(Vec<Vec<[u32; 2]>>);

impl Sink for Counts {
    fn code(&mut self, contexts: &[usize], _: usize, bit: bool) {
        for (t, &c) in self.0.iter_mut().zip(contexts) {
            t[c][usize::from(bit)] += 1;
        }
    }
}

/// A fixed model: one quantized probability per context and fixed weights.
struct Frozen<'a> {
    stretched: &'a [Vec<f32>],
    weights: Vec<f32>,
    rate: f32,
    bits: f64,
}

impl Sink for Frozen<'_> {
    fn code(&mut self, contexts: &[usize], set: usize, bit: bool) {
        let n = self.stretched.len();
        let w = &mut self.weights[set * n..(set + 1) * n];
        let dot: f32 = contexts
            .iter()
            .zip(self.stretched)
            .zip(w.iter())
            .map(|((&c, t), wi)| wi * t[c])
            .sum();
        let p = squash(dot).clamp(1.0 / 4096.0, 1.0 - 1.0 / 4096.0);
        self.bits -= f64::from(if bit { p } else { 1.0 - p }).log2();
        if self.rate > 0.0 {
            let err = (f32::from(u8::from(bit)) - p) * self.rate;
            for ((wi, &c), t) in w.iter_mut().zip(contexts).zip(self.stretched) {
                *wi += err * t[c];
            }
        }
    }
}

/// Size of a semi-static model: the code length and the header size.
/// One pass counts the bits of each context over the full table.
/// The header keeps a 12-bit probability for each context with TB_PRUNE bits or more.
/// The header size is the zstd size of these values and of a presence bitmap.
/// A second pass trains the mixer weights on a sample of the blocks.
/// A third pass finds the code length with the fixed model.
fn semi_static<W: Walk>(walk: &W, words: usize, block_words: usize) -> (f64, usize) {
    let sizes = walk.sizes();
    let blocks = words.div_ceil(block_words);
    let parts = per_block(walk, words, block_words, 0..blocks, || {
        Counts(sizes.iter().map(|&n| vec![[0; 2]; n]).collect())
    });
    let mut counts = Counts(sizes.iter().map(|&n| vec![[0; 2]; n]).collect());
    for part in parts {
        for (t, u) in counts.0.iter_mut().zip(part.0) {
            for (a, b) in t.iter_mut().zip(u) {
                a[0] += b[0];
                a[1] += b[1];
            }
        }
    }
    let prune = setting("TB_PRUNE", 1u32);
    let mut header = Vec::new();
    let stretched: Vec<Vec<f32>> = counts
        .0
        .iter()
        .map(|t| {
            let mut present = vec![0u8; t.len().div_ceil(8)];
            let mut probs = Vec::new();
            let out = t
                .iter()
                .enumerate()
                .map(|(i, &[n0, n1])| {
                    if n0 + n1 < prune {
                        return 0.0;
                    }
                    let p = (f64::from(n1) + 0.4) / (f64::from(n0 + n1) + 0.8);
                    let q = (p * 4096.0).round().clamp(1.0, 4095.0) as u16;
                    present[i / 8] |= 1 << (i % 8);
                    probs.extend_from_slice(&q.to_le_bytes());
                    stretch(f32::from(q) / 4096.0)
                })
                .collect();
            header.extend_from_slice(&present);
            header.extend_from_slice(&probs);
            out
        })
        .collect();
    let used = zstd::bulk::compress(&header, 19).map_or(header.len(), |h| h.len());
    let n = sizes.len();
    let mut trainer = Frozen {
        stretched: &stretched,
        weights: vec![1.0 / n as f32; walk.sets() * n],
        rate: setting("TB_STATIC_RATE", 0.002),
        bits: 0.0,
    };
    let sample = setting("TB_SAMPLE", 8);
    for _ in 0..setting("TB_EPOCHS", 2) {
        for b in (0..blocks).step_by(sample) {
            let start = b * block_words;
            walk.walk(start, (start + block_words).min(words), &mut trainer);
        }
    }
    let weights = trainer.weights;
    let bits = per_block(walk, words, block_words, 0..blocks, || Frozen {
        stretched: &stretched,
        weights: weights.clone(),
        rate: 0.0,
        bits: 0.0,
    })
    .iter()
    .map(|f| f.bits)
    .sum();
    (bits, used)
}

/// Value at (file + df, rank + dr) of word cw - dw, if the block has it.
fn near<T: Copy>(
    v: &[T],
    none: T,
    start: usize,
    (cw, bk): (usize, usize),
    (dw, df, dr): (usize, i32, i32),
) -> T {
    let (f, r) = ((bk % 8) as i32 + df, (bk / 8) as i32 + dr);
    if cw < start + dw || !(0..8).contains(&f) || !(0..8).contains(&r) {
        return none;
    }
    v[(cw - dw) * 64 + (r * 8 + f) as usize]
}

const WDL_NODES: usize = 6;

/// Read a number from an environment variable, or use the default.
fn setting<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct WdlWalk<'a> {
    wdl: &'a [u8],
    black_start: usize,
    /// Skeleton squares of each stored word.
    squares: &'a [[u8; 5]],
    /// Colour and role of each skeleton piece.
    pieces: &'a [Piece],
}

fn skeletons(layout: &Layout, words: usize) -> Vec<[u8; 5]> {
    let m = layout.material.skeleton_len();
    (0..words)
        .map(|cw| {
            let (_, sq) = layout.material.decode(layout.full_word(cw));
            let mut out = [0u8; 5];
            out[..m].copy_from_slice(&sq[..m]);
            out
        })
        .collect()
}

/// True if a piece of this role on `from` attacks `to`, with the blockers in `occupied`.
fn attacks(role: Role, from: u8, to: u8, occupied: u64) -> bool {
    let (df, dr) = (
        i32::from(to % 8) - i32::from(from % 8),
        i32::from(to / 8) - i32::from(from / 8),
    );
    let (straight, diagonal) = (df == 0 || dr == 0, df.abs() == dr.abs());
    let slides = match role {
        Role::King => return df.abs().max(dr.abs()) == 1,
        Role::Knight => return df.abs() * dr.abs() == 2,
        Role::Queen => straight || diagonal,
        Role::Rook => straight,
        Role::Bishop => diagonal,
    };
    if !slides || (df == 0 && dr == 0) {
        return false;
    }
    let step = df.signum() + 8 * dr.signum();
    let mut sq = i32::from(from) + step;
    while sq != i32::from(to) {
        if occupied >> sq & 1 == 1 {
            return false;
        }
        sq += step;
    }
    true
}

fn distance(a: u8, b: u8) -> i32 {
    (i32::from(a % 8) - i32::from(b % 8))
        .abs()
        .max((i32::from(a / 8) - i32::from(b / 8)).abs())
}

const GEOMETRY: usize = 16 * 16 * 16 * 3;

/// Geometry context: the white checks, the loose pieces and the king distance.
/// A loose piece is a piece that the other side can take without a recapture.
fn geometry(pieces: &[Piece], squares: &[u8], bk: u8) -> usize {
    let occupied = squares.iter().fold(1u64 << bk, |o, &s| o | 1 << s);
    let attackers = |white: bool, target: usize| {
        let to = if target == usize::MAX {
            bk
        } else {
            squares[target]
        };
        let king = !white && target != usize::MAX && distance(bk, to) == 1;
        king || pieces.iter().zip(squares).enumerate().any(|(j, (p, &sq))| {
            j != target && p.white == white && attacks(p.role, sq, to, occupied)
        })
    };
    let (mut check, mut white_loose, mut black_loose) = (0, 0, 0);
    for (j, p) in pieces.iter().enumerate().skip(1) {
        let bit = 1 << (j - 1);
        if p.white && attacks(p.role, squares[j], bk, occupied) {
            check |= bit;
        }
        if attackers(!p.white, j) && !attackers(p.white, j) {
            if p.white {
                white_loose |= bit;
            } else {
                black_loose |= bit;
            }
        }
    }
    let kings = (distance(squares[0], bk) - 2).clamp(0, 2) as usize;
    ((check * 16 + white_loose) * 16 + black_loose) * 3 + kings
}

impl Walk for WdlWalk<'_> {
    fn sizes(&self) -> Vec<usize> {
        [
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
        ]
        .map(|n| n * WDL_NODES)
        .to_vec()
    }

    fn sets(&self) -> usize {
        WDL_NODES * 5
    }

    fn walk<S: Sink>(&self, start: usize, end: usize, m: &mut S) {
        let (wdl, black_start) = (self.wdl, self.black_start);
        let mut c = [0usize; 14];
        let (mut previous, mut run_prev, mut run_w1, mut run_left) =
            (usize::from(UNKNOWN), 0, 0, 0);
        for cw in start..end {
            let turn = usize::from(cw >= black_start) * 3;
            let squares = &self.squares[cw][..self.pieces.len()];
            for bk in 0..64 {
                let v = wdl[cw * 64 + bk];
                if v == UNKNOWN {
                    continue;
                }
                let geo = geometry(self.pieces, squares, bk as u8);
                let at =
                    |dw, df, dr| usize::from(near(wdl, UNKNOWN, start, (cw, bk), (dw, df, dr)));
                let (left, up, ul, ur) = (at(0, -1, 0), at(0, 0, -1), at(0, -1, -1), at(0, 1, -1));
                let (w1, w8, w64, w512, w4096) = (
                    at(1, 0, 0),
                    at(8, 0, 0),
                    at(64, 0, 0),
                    at(512, 0, 0),
                    at(4096, 0, 0),
                );
                let (w1l, w1u) = (at(1, -1, 0), at(1, 0, -1));
                let board = left * 5 + up;
                let word = board * 25 + w1 * 5 + w8;
                let base = [
                    0,
                    board,
                    word,
                    word * 25 + ul * 5 + ur,
                    word * 25 + w64 * 5 + w512,
                    (word * 25 + ul * 5 + ur) * 5 + w64,
                    (((w1 * 5 + w8) * 5 + w64) * 5 + w512) * 5 + w4096,
                    (((board * 5 + w1) * 5 + w1l) * 5) + w1u,
                    (w1 * 8 + run_bucket(run_w1)) * 5 + left,
                    ((board * 8 + run_bucket(run_left)) * 5) + w1,
                    (previous * 8 + run_bucket(run_prev)) * 5 + up,
                    geo,
                    geo * 25 + board,
                    geo * 25 + w1 * 5 + w8,
                ];
                let mut node = |n: usize, bit: bool| {
                    for (ci, b) in c.iter_mut().zip(base) {
                        *ci = b * WDL_NODES + n + turn;
                    }
                    m.code(&c, (n + turn) * 5 + left, bit);
                };
                node(0, v == 0);
                if v != 0 {
                    node(1, v == 2);
                    if v != 2 && turn != 0 {
                        node(2, v == 3);
                    }
                }
                let v = usize::from(v);
                run_prev = if v == previous { run_prev + 1 } else { 0 };
                run_w1 = if v == w1 { run_w1 + 1 } else { 0 };
                run_left = if v == left { run_left + 1 } else { 0 };
                previous = v;
            }
        }
    }
}

/// Bucket of a run length: 0, 1, 2-3, 4-7, up to 64 and more.
fn run_bucket(run: u32) -> usize {
    (run + 1).ilog2().min(7) as usize
}

const RANK_NODES: usize = 160;

/// Context value of a neighbour rank relative to the predicted rank.
fn rel(x: u16, pred: u16) -> usize {
    if x == NO_RANK {
        17
    } else {
        (i32::from(x) - i32::from(pred)).clamp(-8, 8) as usize + 8
    }
}

struct RankWalk<'a> {
    wdl: &'a [u8],
    rank: &'a [u16],
    squares: &'a [[u8; 5]],
    pieces: &'a [Piece],
}

/// Small geometry context: a check, a loose white piece, a loose black piece and the king distance.
fn small_geometry(geo: usize) -> usize {
    let (kings, black, white, check) = (geo % 3, geo / 3 % 16, geo / 48 % 16, geo / 768);
    ((usize::from(check > 0) * 2 + usize::from(white > 0)) * 2 + usize::from(black > 0)) * 3 + kings
}

impl Walk for RankWalk<'_> {
    fn sizes(&self) -> Vec<usize> {
        [8, 2056, 648, 11664, 11664, 9252, 11664, 2 * GEOMETRY, 15552]
            .map(|n| n * RANK_NODES)
            .to_vec()
    }

    fn sets(&self) -> usize {
        RANK_NODES * 2
    }

    fn walk<S: Sink>(&self, start: usize, end: usize, m: &mut S) {
        let (wdl, rank) = (self.wdl, self.rank);
        let mut c = [0usize; 9];
        for cw in start..end {
            let squares = &self.squares[cw][..self.pieces.len()];
            for bk in 0..64 {
                let r = rank[cw * 64 + bk];
                if r == NO_RANK {
                    continue;
                }
                let geo = geometry(self.pieces, squares, bk as u8);
                let win = usize::from(wdl[cw * 64 + bk] == 1);
                let at = |dw, df, dr| near(rank, NO_RANK, start, (cw, bk), (dw, df, dr));
                let (left, up, ul, ur) = (at(0, -1, 0), at(0, 0, -1), at(0, -1, -1), at(0, 1, -1));
                let (w1, w8, w64) = (at(1, 0, 0), at(8, 0, 0), at(64, 0, 0));
                let (kind, pred) = [left, up, w1]
                    .iter()
                    .position(|&x| x != NO_RANK)
                    .map_or((3, 0), |i| (i, [left, up, w1][i]));
                let p = usize::from(pred);
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
                        *ci = b * RANK_NODES + n;
                    }
                    m.code(&c, n * 2 + win, bit);
                };
                let d = i32::from(r) - i32::from(pred);
                node(0, d == 0);
                if d == 0 {
                    continue;
                }
                let neg = usize::from(d < 0) * 80;
                node(1, d < 0);
                let mag = d.unsigned_abs();
                let k = mag.ilog2() as usize;
                for j in 0..k {
                    node(2 + j + neg, true);
                }
                if k < 7 {
                    node(2 + k + neg, false);
                }
                for i in (0..k).rev() {
                    node(10 + k * 8 + i + neg, mag >> i & 1 == 1);
                }
            }
        }
    }
}

fn main() -> Result<(), String> {
    let mut report = Vec::new();
    for input in std::env::args().skip(1).map(PathBuf::from) {
        let t = Instant::now();
        let header = PackedTable::open(&input, false)?.header;
        let (table, _) = packed::load_table(&input, 16)?;
        let stored = Canonical::from_table(&table);
        let layout = Layout::new(&table.material);
        let (wdl, rank) = values(&layout, &stored);
        let words = stored.records.len();
        let known = wdl.iter().filter(|&&v| v != UNKNOWN).count();
        let ranked = rank.iter().filter(|&&r| r != NO_RANK).count();
        let squares = skeletons(&layout, words);
        let pieces = &table.material.skeleton;
        let wdl_walk = WdlWalk {
            wdl: &wdl,
            black_start: words / 2,
            squares: &squares,
            pieces,
        };
        let rank_walk = RankWalk {
            wdl: &wdl,
            rank: &rank,
            squares: &squares,
            pieces,
        };
        let (wdl_block, rank_block) = (
            setting("TB_WDL_BLOCK_WORDS", BLOCK / 16),
            setting("TB_RANK_BLOCK_WORDS", BLOCK / 64),
        );
        let t_mix = Instant::now();
        let wdl_mix = adaptive(&wdl_walk, words, wdl_block);
        let wdl_mix_s = t_mix.elapsed().as_secs_f64();
        let t_mix = Instant::now();
        let rank_mix = adaptive(&rank_walk, words, rank_block);
        let rank_mix_s = t_mix.elapsed().as_secs_f64();
        let (static_wdl, static_rank) = if setting("TB_STATIC", 1) == 1 {
            (
                semi_static(&wdl_walk, words, setting("TB_STATIC_WDL_BLOCK_WORDS", 256)),
                semi_static(&rank_walk, words, setting("TB_STATIC_RANK_BLOCK_WORDS", 64)),
            )
        } else {
            ((0.0, 0), (0.0, 0))
        };
        let (w, r) = if setting("TB_MIXER_ONLY", 0) == 1 {
            (Vec::new(), Vec::new())
        } else {
            std::thread::scope(|s| {
                let w = s.spawn(|| wdl_bits(&wdl, words));
                let r = s.spawn(|| rank_bits(&wdl, &rank, words));
                (w.join().expect("wdl"), r.join().expect("rank"))
            })
        };
        let bytes = |v: Vec<(&str, f64)>| -> serde_json::Value {
            v.into_iter()
                .map(|(n, b)| (n.to_string(), serde_json::json!((b / 8.0).round())))
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        let row = serde_json::json!({
            "material": table.material.name(), "stored_positions": words * 64,
            "coded_wdl_positions": known, "coded_rank_positions": ranked,
            "file_sections": header.sections.iter()
                .map(|s| (s.name.clone(), s.compressed_bytes)).collect::<Vec<_>>(),
            "wdl_bytes": bytes(w), "rank_bytes": bytes(r),
            "wdl_mixing_bytes": (wdl_mix / 8.0).round(), "rank_mixing_bytes": (rank_mix / 8.0).round(),
            "wdl_mixing_seconds": wdl_mix_s, "rank_mixing_seconds": rank_mix_s,
            "wdl_static_bytes": (static_wdl.0 / 8.0).round(), "wdl_static_header_bytes": static_wdl.1,
            "rank_static_bytes": (static_rank.0 / 8.0).round(), "rank_static_header_bytes": static_rank.1,
            "seconds": t.elapsed().as_secs_f64(),
        });
        eprintln!("{row}");
        report.push(row);
    }
    println!(
        "{}",
        serde_json::json!({"tool": "tb_entropy", "tables": report})
    );
    Ok(())
}
