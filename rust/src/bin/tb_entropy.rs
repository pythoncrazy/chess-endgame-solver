//! Compare the `.tbz` section sizes with adaptive code lengths.
//! The tool codes the stored WDL and rank values with different contexts.
//!
//! The models skip positions that a reader can find without the file:
//! illegal positions and positions that are not stored. Each model starts
//! again at each block, as a block coder must. The result is the size of an
//! ideal arithmetic coder with that model, without block headers.
use std::{path::PathBuf, time::Instant};
use tablebase_core::packed::{self, BLOCK, Canonical, Layout, PackedTable};

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
        let (w, r) = std::thread::scope(|s| {
            let w = s.spawn(|| wdl_bits(&wdl, words));
            let r = s.spawn(|| rank_bits(&wdl, &rank, words));
            (w.join().expect("wdl"), r.join().expect("rank"))
        });
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
