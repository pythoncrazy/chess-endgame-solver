//! Compressed pawnless tables, format v4 (`TBPAWNZ2`); v3 (`TBPAWNZ1`) is
//! still read. Specification: `TBZ_FORMAT.md`.
//!
//! Each position is stored once per symmetry class. Its stored image has the
//! white king in the a1-d1-d4 triangle and, among the images that do, the
//! smallest index (the transpose decides for a white king on the long
//! diagonal; identical pieces take ascending squares). Illegal positions and
//! non-stored images are "don't care": they repeat the previous stored value,
//! which costs almost nothing once compressed. Readers decide legality with
//! `pawnless::statics`, so the file never encodes it.
//!
//! Stored index: `((turn * 10 + triangle(white king)) * 64^(m-1) + other
//! skeleton squares) * 64 + black king`. Two sections, each a block index
//! followed by independent zstd or LZMA frames of `BLOCK` decompressed bytes:
//! `wdl` (2 bits per position: 0 draw, 1 win, 2 loss, 3 black-to-move win of
//! rank 1) and `rank` (proof rank of decisive positions, u8 or u16). By
//! default only white-to-move ranks are stored; black-to-move ranks follow
//! exactly from the white-to-move children. A probe decompresses at most one
//! block per section, plus the children's blocks for a black-to-move rank.
use crate::Result;
use crate::pawnless::{
    Counts, DIGEST_TREE, Geometry, MAX_SKELETON, Material, Symmetry, Table, UNRESOLVED,
    payload_digest, statics,
};
use lru::LruCache;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Write},
    num::NonZeroUsize,
    ops::Range,
    os::unix::fs::FileExt,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

pub const MAGIC: &[u8; 8] = b"TBPAWNZ2";
/// Format v3: both rank sides, zstd only, no WDL code 3 (read-only).
pub const MAGIC_V3: &[u8; 8] = b"TBPAWNZ1";
/// Decompressed bytes per block (the last block of a section may be shorter).
pub const BLOCK: usize = 1 << 18;
/// White-king squares of the stored domain, in index order.
pub const TRIANGLE: [u8; 10] = [0, 1, 2, 3, 9, 10, 11, 18, 19, 27];
const TRANSPOSE: Symmetry = Symmetry {
    transpose: true,
    mirror_files: false,
    mirror_ranks: false,
};
/// Squares on or below the a1-h8 diagonal (file >= rank): the stored black
/// king squares when the transpose leaves the skeleton unchanged.
const LOWER_RIGHT: u64 = 0x80c0_e0f0_f8fc_feff;
const BLOCK_CACHE: usize = 256;
/// Decompressed blocks keyed by (rank section, block number).
type BlockCache = LruCache<(bool, usize), Arc<Vec<u8>>>;

const TRIANGLE_LOOKUP: [Option<usize>; 64] = {
    let mut table = [None; 64];
    let mut i = 0;
    while i < TRIANGLE.len() {
        table[TRIANGLE[i] as usize] = Some(i);
        i += 1;
    }
    table
};

const TO_TRIANGLE: [Symmetry; 64] = {
    let mut table = [Symmetry {
        transpose: false,
        mirror_files: false,
        mirror_ranks: false,
    }; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let mut b = 0u8;
        while b < 8 {
            let g = Symmetry {
                transpose: b & 1 != 0,
                mirror_files: b & 2 != 0,
                mirror_ranks: b & 4 != 0,
            };
            let mut s = sq as u8;
            if g.transpose {
                s = ((s & 7) << 3) | (s >> 3);
            }
            if g.mirror_files {
                s ^= 7;
            }
            if g.mirror_ranks {
                s ^= 56;
            }
            if TRIANGLE_LOOKUP[s as usize].is_some() {
                table[sq] = g;
                break;
            }
            b += 1;
        }
        sq += 1;
    }
    table
};

fn triangle_index(square: u8) -> Option<usize> {
    TRIANGLE_LOOKUP[usize::from(square)]
}

fn on_diagonal(square: u8) -> bool {
    square % 8 == square / 8
}

/// The symmetry that moves `square` into the triangle (the identity or the
/// transpose both qualify on the diagonal; the identity form is returned).
fn to_triangle(square: u8) -> Symmetry {
    TO_TRIANGLE[usize::from(square)]
}

/// Where a full-layout word's values live in the stored stream.
#[derive(Clone, Copy, Debug)]
struct Image {
    word: usize,
    /// Square map: `g`, then the transpose when `transpose` is set.
    g: Symmetry,
    transpose: bool,
    /// The transpose fixes the skeleton: each black-king square takes the
    /// smaller of its two images.
    per_square: bool,
}

impl Image {
    fn square(&self, s: u8) -> u8 {
        let mut t = self.g.square(s);
        if self.transpose {
            t = TRANSPOSE.square(t);
        }
        if self.per_square && LOWER_RIGHT >> t & 1 == 0 {
            t = TRANSPOSE.square(t);
        }
        t
    }

    /// Map a bitboard from the stored word (`self.word`) back to the full word.
    fn bitboard(&self, mut y: u64) -> u64 {
        if self.per_square {
            let lower = y & LOWER_RIGHT;
            y = lower | TRANSPOSE.bitboard(lower);
        }
        if self.transpose {
            y = TRANSPOSE.bitboard(y);
        }
        self.g.inverse_bitboard(y)
    }
}

/// Stored-domain geometry of one material.
#[derive(Clone, Debug)]
pub struct Layout {
    pub material: Material,
    m: usize,
    /// 64^(m-1): words per (turn, white-king) slice.
    per: usize,
    /// Skeleton ranges of identical pieces (length at least two).
    groups: Vec<Range<usize>>,
}

impl Layout {
    pub fn new(material: &Material) -> Layout {
        let m = material.skeleton_len();
        let mut groups = Vec::new();
        let mut start = 1;
        for i in 2..=m {
            if i == m || material.skeleton[i] != material.skeleton[start] {
                if i - start >= 2 {
                    groups.push(start..i);
                }
                start = i;
            }
        }
        Layout {
            material: material.clone(),
            m,
            per: 64usize.pow(m as u32 - 1),
            groups,
        }
    }

    /// Stored words: two turns times ten white-king squares times 64^(m-1).
    pub fn words(&self) -> usize {
        20 * self.per
    }

    /// Full-layout word of stored word `cw`.
    pub fn full_word(&self, cw: usize) -> usize {
        let (head, rest) = (cw / self.per, cw % self.per);
        let (turn, tri) = (head / 10, head % 10);
        (turn * 64 + usize::from(TRIANGLE[tri])) * self.per + rest
    }

    fn stored_word(&self, black: bool, squares: &[u8]) -> usize {
        let tri = triangle_index(squares[0]).expect("white king in the triangle");
        let rest = squares[1..self.m]
            .iter()
            .fold(0usize, |i, &s| i * 64 + usize::from(s));
        (usize::from(black) * 10 + tri) * self.per + rest
    }

    fn sort_groups(&self, squares: &mut [u8]) {
        for g in &self.groups {
            squares[g.clone()].sort_unstable();
        }
    }

    fn mapped(&self, g: Symmetry, squares: &[u8]) -> [u8; MAX_SKELETON] {
        let mut out = [0u8; MAX_SKELETON];
        for (o, &s) in out.iter_mut().zip(&squares[..self.m]) {
            *o = g.square(s);
        }
        self.sort_groups(&mut out[..self.m]);
        out
    }

    /// Black-king squares whose positions are stored in word `squares`
    /// (white king already in the triangle), before legality.
    fn stored_mask(&self, squares: &[u8]) -> u64 {
        let s = &squares[..self.m];
        if self
            .groups
            .iter()
            .any(|g| s[g.clone()].windows(2).any(|w| w[0] >= w[1]))
        {
            return 0;
        }
        if !on_diagonal(s[0]) {
            return u64::MAX;
        }
        let t = self.mapped(TRANSPOSE, s);
        match t[..self.m].cmp(s) {
            std::cmp::Ordering::Less => 0,
            std::cmp::Ordering::Greater => u64::MAX,
            std::cmp::Ordering::Equal => LOWER_RIGHT,
        }
    }

    fn image(&self, black: bool, squares: &[u8]) -> Image {
        let g = to_triangle(squares[0]);
        let s1 = self.mapped(g, squares);
        let mut image = Image {
            word: 0,
            g,
            transpose: false,
            per_square: false,
        };
        let mut stored = s1;
        if on_diagonal(s1[0]) {
            let s2 = self.mapped(TRANSPOSE, &s1);
            match s2[..self.m].cmp(&s1[..self.m]) {
                std::cmp::Ordering::Less => {
                    image.transpose = true;
                    stored = s2;
                }
                std::cmp::Ordering::Equal => image.per_square = true,
                std::cmp::Ordering::Greater => {}
            }
        }
        image.word = self.stored_word(black, &stored[..self.m]);
        image
    }

    /// Black-king squares stored in full-layout word `word` (before legality):
    /// zero unless the white king is in the triangle.
    pub fn stored_in_word(&self, word: usize) -> u64 {
        let (_, squares) = self.material.decode(word);
        if triangle_index(squares[0]).is_none() {
            return 0;
        }
        self.stored_mask(&squares)
    }

    /// Stored position index of a position (white king anywhere).
    pub fn position(&self, black: bool, squares: &[u8], black_king: u8) -> usize {
        let image = self.image(black, squares);
        image.word * 64 + usize::from(image.square(black_king))
    }
}

/// A solved table restricted to stored words (white king in the triangle).
#[derive(Clone, Debug)]
pub struct Canonical {
    pub material: Material,
    pub rounds: u32,
    /// Per stored word: `[win | illegal, loss | illegal]`.
    pub records: Vec<[u64; 2]>,
    /// Proof rank per stored position (`word * 64 + black king`), UNRESOLVED
    /// when undecided; empty for tables without ranks.
    pub rank: Vec<u16>,
    pub counts: Option<[Counts; 2]>,
}

impl Canonical {
    pub fn from_table(table: &Table) -> Canonical {
        let layout = Layout::new(&table.material);
        let words = layout.words();
        let mut records = Vec::with_capacity(words);
        let mut rank = Vec::with_capacity(if table.rank.is_empty() { 0 } else { words * 64 });
        for cw in 0..words {
            let w = layout.full_word(cw);
            let illegal = !table.valid[w];
            records.push([table.win[w] | illegal, table.loss[w] | illegal]);
            if !table.rank.is_empty() {
                rank.extend_from_slice(&table.rank[w * 64..w * 64 + 64]);
            }
        }
        Canonical {
            material: table.material.clone(),
            rounds: table.rounds,
            records,
            rank,
            counts: Some(table.counts()),
        }
    }

    fn rank_width(&self) -> usize {
        match (self.rank.is_empty(), self.rounds < 255) {
            (true, _) => 0,
            (false, true) => 1,
            (false, false) => 2,
        }
    }
}

/// Block compression of a section.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Codec {
    /// One zstd frame per block.
    #[default]
    Zstd,
    /// One LZMA1 `.lzma` ("alone") stream per block (preset 9e, lc 4): about
    /// 10% smaller rank sections than zstd 19, about 6x slower cold probes.
    Lzma,
}

/// Sides to move whose proof ranks are stored.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RankSides {
    #[default]
    Both,
    /// White to move only; black-to-move ranks follow from the children.
    White,
}

/// How `write` encodes a table.
#[derive(Clone, Copy, Debug)]
pub struct WriteOptions {
    /// zstd level (LZMA always uses preset 9e).
    pub level: i32,
    pub wdl_codec: Codec,
    pub rank_codec: Codec,
    pub rank_sides: RankSides,
}

impl Default for WriteOptions {
    fn default() -> Self {
        WriteOptions {
            level: 19,
            wdl_codec: Codec::Zstd,
            rank_codec: Codec::Zstd,
            rank_sides: RankSides::White,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    /// Offset of the block index from the payload start; frames follow it.
    pub offset: u64,
    pub blocks: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    #[serde(default)]
    pub codec: Codec,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackedHeader {
    pub format: String,
    pub material: String,
    pub layout: String,
    pub stored_words: usize,
    pub rounds: u32,
    pub rank_width: usize,
    pub block_bytes: usize,
    pub zstd_level: i32,
    pub sections: Vec<Section>,
    /// SHA-256 tree digest (`DIGEST_TREE`) of every byte after the header.
    pub payload_sha256: String,
    pub digest_scheme: String,
    /// SHA-256 over the SHA-256 of each decompressed block, sections in
    /// order: identifies the stored values independently of compression.
    pub content_sha256: String,
    pub generator: String,
    #[serde(default)]
    pub counts: Option<[Counts; 2]>,
    #[serde(default)]
    pub rank_sides: RankSides,
}

/// Run `f` on indices `0..n` with dynamic scheduling over `threads` workers.
fn parallel<T: Send>(
    n: usize,
    threads: usize,
    f: impl Fn(usize) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let failure = Mutex::new(None::<String>);
    std::thread::scope(|scope| {
        for _ in 0..threads.clamp(1, n.max(1)) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    match f(i) {
                        Ok(v) => *slots[i].lock().expect("slot") = Some(v),
                        Err(e) => {
                            failure.lock().expect("failure").get_or_insert(e);
                            next.store(n, Ordering::Relaxed);
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = failure.into_inner().expect("failure") {
        return Err(e);
    }
    Ok(slots
        .into_iter()
        .map(|s| s.into_inner().expect("slot").expect("filled"))
        .collect())
}

/// Stored, legal positions of stored word `cw`.
fn stored_bits(layout: &Layout, c: &Canonical, cw: usize) -> u64 {
    let [win, loss] = c.records[cw];
    let valid = !(win & loss);
    if valid == 0 {
        return 0;
    }
    let (_, squares) = layout.material.decode(layout.full_word(cw));
    layout.stored_mask(&squares) & valid
}

/// Black-to-move stored words start here in the stored index.
fn black_start(layout: &Layout) -> usize {
    10 * layout.per
}

/// Decompressed WDL block `b`: 4096 stored words, four positions per byte.
/// With white-only ranks, black-to-move wins of rank 1 take code 3.
fn wdl_block(layout: &Layout, c: &Canonical, b: usize, sides: RankSides) -> Vec<u8> {
    let words_per_block = BLOCK / 16;
    let range = b * words_per_block..((b + 1) * words_per_block).min(c.records.len());
    let mut out = vec![0u8; range.len() * 16];
    let mut prev = 0u8;
    let mark_rank1 = sides == RankSides::White && !c.rank.is_empty();
    for (i, cw) in range.enumerate() {
        let [win, loss] = c.records[cw];
        let stored = stored_bits(layout, c, cw);
        if stored == 0 {
            out[i * 16..(i + 1) * 16].fill(prev * 0x55);
            continue;
        }
        let black = cw >= black_start(layout);
        for bk in 0..64 {
            if stored >> bk & 1 != 0 {
                prev = ((win >> bk & 1) as u8) | (((loss >> bk & 1) as u8) << 1);
                if prev == 1 && black && mark_rank1 && c.rank[cw * 64 + bk] == 1 {
                    prev = 3;
                }
            }
            let p = i * 64 + bk;
            out[p / 4] |= prev << (2 * (p % 4));
        }
    }
    out
}

/// Decompressed rank block `b`.
fn rank_block(layout: &Layout, c: &Canonical, b: usize, width: usize, words: usize) -> Vec<u8> {
    let words_per_block = BLOCK / (64 * width);
    let range = b * words_per_block..((b + 1) * words_per_block).min(words);
    let mut out = vec![0u8; range.len() * 64 * width];
    let mut prev = 0u16;
    for (i, cw) in range.enumerate() {
        let [win, loss] = c.records[cw];
        let decisive = stored_bits(layout, c, cw) & (win | loss);
        if decisive == 0 {
            if width == 1 {
                out[i * 64..(i + 1) * 64].fill(prev as u8);
            } else {
                let bytes = prev.to_le_bytes();
                for dst in out[i * 128..(i + 1) * 128].as_chunks_mut::<2>().0 {
                    dst.copy_from_slice(&bytes);
                }
            }
            continue;
        }
        for bk in 0..64 {
            if decisive >> bk & 1 != 0 {
                prev = c.rank[cw * 64 + bk];
            }
            let p = (i * 64 + bk) * width;
            if width == 1 {
                out[p] = prev as u8;
            } else {
                out[p..p + 2].copy_from_slice(&prev.to_le_bytes());
            }
        }
    }
    out
}

fn compress_block(codec: Codec, level: i32, raw: &[u8]) -> Result<Vec<u8>> {
    match codec {
        Codec::Zstd => {
            thread_local! {
                static ZSTD: std::cell::RefCell<Option<(i32, zstd::bulk::Compressor<'static>)>> =
                    const { std::cell::RefCell::new(None) };
            }
            ZSTD.with(|cell| {
                let mut slot = cell.borrow_mut();
                if slot.as_ref().is_none_or(|(l, _)| *l != level) {
                    let compressor =
                        zstd::bulk::Compressor::new(level).map_err(|e| e.to_string())?;
                    *slot = Some((level, compressor));
                }
                let (_, compressor) = slot.as_mut().expect("compressor");
                compressor.compress(raw).map_err(|e| e.to_string())
            })
        }
        Codec::Lzma => {
            // Preset 9e with lc = 4; the dictionary never needs to exceed a block.
            let mut opts =
                xz2::stream::LzmaOptions::new_preset(9 | 0x8000_0000).map_err(|e| e.to_string())?;
            opts.literal_context_bits(4)
                .nice_len(273)
                .dict_size(BLOCK as u32);
            let stream = xz2::stream::Stream::new_lzma_encoder(&opts).map_err(|e| e.to_string())?;
            let mut encoder = xz2::write::XzEncoder::new_stream(Vec::new(), stream);
            encoder.write_all(raw).map_err(|e| e.to_string())?;
            encoder.finish().map_err(|e| e.to_string())
        }
    }
}

fn decompress_block(codec: Codec, frame: &[u8], expect: usize) -> Result<Vec<u8>> {
    match codec {
        Codec::Zstd => zstd::bulk::decompress(frame, expect).map_err(|e| e.to_string()),
        Codec::Lzma => {
            let stream =
                xz2::stream::Stream::new_lzma_decoder(u64::MAX).map_err(|e| e.to_string())?;
            let mut out = Vec::with_capacity(expect);
            xz2::read::XzDecoder::new_stream(frame, stream)
                .take(expect as u64 + 1)
                .read_to_end(&mut out)
                .map_err(|e| e.to_string())?;
            Ok(out)
        }
    }
}

/// Compress a stored table into a v4 file image.
pub fn write(
    c: &Canonical,
    mut out: impl Write,
    generator: &str,
    options: WriteOptions,
    threads: usize,
) -> Result<PackedHeader> {
    let layout = Layout::new(&c.material);
    if c.records.len() != layout.words() {
        return Err("stored word count disagrees with the material".into());
    }
    let width = c.rank_width();
    if width == 1 && c.rank.iter().any(|&r| r != UNRESOLVED && r >= 255) {
        return Err("rank exceeds the u8 representation".into());
    }
    let sides = if width == 0 {
        RankSides::Both
    } else {
        options.rank_sides
    };
    let rank_words = match sides {
        RankSides::Both => c.records.len(),
        RankSides::White => black_start(&layout),
    };
    let mut sections = vec![("wdl", c.records.len() * 16, options.wdl_codec)];
    if width > 0 {
        sections.push(("rank", rank_words * 64 * width, options.rank_codec));
    }
    let mut payload = Vec::new();
    let mut infos = Vec::new();
    let mut content = Sha256::new();
    for (name, raw_bytes, codec) in sections {
        let blocks = raw_bytes.div_ceil(BLOCK);
        let frames = parallel(blocks, threads, |b| {
            let raw = if name == "wdl" {
                wdl_block(&layout, c, b, sides)
            } else {
                rank_block(&layout, c, b, width, rank_words)
            };
            let digest: [u8; 32] = Sha256::digest(&raw).into();
            Ok((digest, compress_block(codec, options.level, &raw)?))
        })?;
        let offset = payload.len() as u64;
        let mut at = 0u64;
        for (_, frame) in &frames {
            payload.extend_from_slice(&at.to_le_bytes());
            at += frame.len() as u64;
        }
        payload.extend_from_slice(&at.to_le_bytes());
        for (digest, frame) in &frames {
            content.update(digest);
            payload.extend_from_slice(frame);
        }
        infos.push(Section {
            name: name.into(),
            offset,
            blocks: blocks as u64,
            raw_bytes: raw_bytes as u64,
            compressed_bytes: at,
            codec,
        });
    }
    let header = PackedHeader {
        format: "tablebase-pawnless-v4".into(),
        material: c.material.name(),
        layout: "stored once per symmetry class: white king in a1-d1-d4, smallest index \
                 among equivalent images; position = ((turn*10 + triangle(white king)) \
                 * 64^(m-1) + other skeleton squares) * 64 + black king; wdl 2 bits \
                 (0 draw, 1 win, 2 loss, 3 black-to-move win of rank 1 when only white \
                 ranks are stored), rank u8 or u16 for the stored sides; illegal and \
                 non-stored positions repeat the previous value"
            .into(),
        stored_words: c.records.len(),
        rounds: c.rounds,
        rank_width: width,
        block_bytes: BLOCK,
        zstd_level: options.level,
        sections: infos,
        payload_sha256: payload_digest(&payload, DIGEST_TREE)?,
        digest_scheme: DIGEST_TREE.into(),
        content_sha256: format!("{:x}", content.finalize()),
        generator: generator.into(),
        counts: c.counts,
        rank_sides: sides,
    };
    let json = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    let io = |e: std::io::Error| e.to_string();
    out.write_all(MAGIC).map_err(io)?;
    out.write_all(&(json.len() as u64).to_le_bytes())
        .map_err(io)?;
    out.write_all(&json).map_err(io)?;
    out.write_all(&payload).map_err(io)?;
    out.flush().map_err(io)?;
    Ok(header)
}

/// White-to-move children of black-to-move positions, same material.
trait Children {
    /// Legal black-king squares of white-to-move word `word`.
    fn valid(&self, word: usize) -> Result<u64>;
    fn is_loss(&self, word: usize, black_king: u8) -> Result<bool>;
    /// Proof rank of a decisive position.
    fn rank(&self, word: usize, black_king: u8) -> Result<u16>;
}

/// Material after each capturable (white, non-king) skeleton piece is taken.
fn lower_materials(material: &Material) -> Vec<Option<Material>> {
    (0..material.skeleton_len())
        .map(|k| (k > 0 && material.skeleton[k].white).then(|| material.without(k).expect("k > 0")))
        .collect()
}

/// Legal black-king squares after white skeleton piece `k` is captured.
fn lower_valid(geometry: &Geometry, lower: &Material, squares: &[u8], k: usize) -> u64 {
    let mut rest = [0u8; MAX_SKELETON];
    let mut n = 0;
    for (i, &s) in squares.iter().enumerate() {
        if i != k {
            rest[n] = s;
            n += 1;
        }
    }
    statics(geometry, lower, lower.encode(false, &rest[..n])).valid
}

/// Proof ranks of the black-to-move positions `lanes` of word `w` (decisive,
/// `win` marks the wins), from the white-to-move children: a win is one more
/// than its fastest losing child, a loss one more than its slowest child, and
/// captures count as rank 0.
#[allow(clippy::too_many_arguments)]
fn derive_black<C: Children + ?Sized>(
    geometry: &Geometry,
    material: &Material,
    lower: &[Option<Material>],
    kids: &C,
    w: usize,
    lanes: u64,
    win: u64,
) -> Result<[u16; 64]> {
    let mut out = [UNRESOLVED; 64];
    if lanes == 0 {
        return Ok(out);
    }
    let len = material.skeleton_len();
    let (_, mut squares) = material.decode(w);
    let squares = &mut squares[..len];
    let occupied = squares.iter().fold(0u64, |o, &s| o | 1 << s);
    let mut win_min = [u16::MAX; 64];
    let mut loss_max = [0u16; 64];
    let mut moved = 0u64;
    let mut child = |word: usize, legal: u64, to_king: &dyn Fn(u8) -> u8| -> Result<()> {
        let mut bits = legal;
        while bits != 0 {
            let bk = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let at = usize::from(bk);
            let kid = to_king(bk);
            if win >> bk & 1 != 0 {
                if kids.is_loss(word, kid)? {
                    win_min[at] = win_min[at].min(kids.rank(word, kid)?);
                }
            } else {
                loss_max[at] = loss_max[at].max(kids.rank(word, kid)?);
            }
        }
        Ok(())
    };
    // Black king moves and captures.
    let same = material.encode(false, squares);
    let valid = kids.valid(same)?;
    let mut bits = lanes;
    while bits != 0 {
        let bk = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let mut to = geometry.king(bk) & valid;
        while to != 0 {
            let t = to.trailing_zeros() as u8;
            to &= to - 1;
            moved |= 1 << bk;
            child(same, 1 << bk, &|_| t)?;
        }
    }
    for (k, lm) in lower.iter().enumerate() {
        let Some(lm) = lm else { continue };
        let t = squares[k];
        let from = geometry.king(t) & lanes;
        if from != 0 && lower_valid(geometry, lm, squares, k) >> t & 1 != 0 {
            moved |= from;
        }
    }
    // Black piece moves; the black king may block a ray.
    for i in 1..len {
        let piece = material.skeleton[i];
        if piece.white {
            continue;
        }
        let from = squares[i];
        let mut targets = geometry.attacks(piece.role, from, occupied) & !(1 << squares[0]);
        while targets != 0 {
            let to = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            let captured = (1..len).find(|&k| squares[k] == to);
            if captured.is_some_and(|k| !material.skeleton[k].white) {
                continue;
            }
            let mask = lanes & !geometry.between(from, to) & !(1 << to);
            if mask == 0 {
                continue;
            }
            squares[i] = to;
            match captured {
                None => {
                    let word = material.encode(false, squares);
                    let legal = mask & kids.valid(word)?;
                    moved |= legal;
                    child(word, legal, &|bk| bk)?;
                }
                Some(k) => {
                    let lm = lower[k].as_ref().expect("white piece");
                    moved |= mask & lower_valid(geometry, lm, squares, k);
                }
            }
            squares[i] = from;
        }
    }
    let mut bits = lanes;
    while bits != 0 {
        let bk = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        out[bk] = if win >> bk & 1 != 0 {
            match win_min[bk] {
                u16::MAX => return Err("black-to-move win without a losing child".into()),
                r => r + 1,
            }
        } else if moved >> bk & 1 != 0 {
            loss_max[bk] + 1
        } else {
            0
        };
    }
    Ok(out)
}

/// Children read from a decoded table (white-to-move ranks filled in).
struct Decoded<'a> {
    valid: &'a [u64],
    loss: &'a [u64],
    rank: &'a [u16],
}

impl Children for Decoded<'_> {
    fn valid(&self, word: usize) -> Result<u64> {
        Ok(self.valid[word])
    }
    fn is_loss(&self, word: usize, black_king: u8) -> Result<bool> {
        Ok(self.loss[word] >> black_king & 1 != 0)
    }
    fn rank(&self, word: usize, black_king: u8) -> Result<u16> {
        Ok(self.rank[word * 64 + usize::from(black_king)])
    }
}

impl Children for PackedTable {
    fn valid(&self, word: usize) -> Result<u64> {
        Ok(statics(&self.geometry, self.material(), word).valid)
    }
    fn is_loss(&self, word: usize, black_king: u8) -> Result<bool> {
        let (_, squares) = self.material().decode(word);
        let p = self
            .layout
            .position(false, &squares[..self.layout.m], black_king);
        Ok(self.code(p)? == 2)
    }
    fn rank(&self, word: usize, black_king: u8) -> Result<u16> {
        let (_, squares) = self.material().decode(word);
        self.stored_rank(
            self.layout
                .position(false, &squares[..self.layout.m], black_king),
        )
    }
}

struct SectionIndex {
    /// Absolute file offset of the first frame.
    frames: u64,
    offsets: Vec<u64>,
    raw_bytes: usize,
    codec: Codec,
}

/// A v3 or v4 table opened for probing; blocks are read on demand and cached.
pub struct PackedTable {
    pub header: PackedHeader,
    pub layout: Layout,
    geometry: Geometry,
    file: File,
    wdl: SectionIndex,
    rank: Option<SectionIndex>,
    cache: Mutex<BlockCache>,
}

/// WDL and proof rank of a legal position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Value {
    /// 2 win, 0 draw, -2 loss for the side to move.
    pub wdl: i8,
    pub rank: Option<u16>,
}

pub fn is_packed(path: &Path) -> Result<bool> {
    let mut magic = [0u8; 8];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(&magic == MAGIC || &magic == MAGIC_V3)
}

impl PackedTable {
    /// Open a v3 or v4 file. With `authenticate`, the payload digest is checked.
    pub fn open(path: &Path, authenticate: bool) -> Result<PackedTable> {
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        let file = File::open(path).map_err(err)?;
        let mut head = [0u8; 16];
        file.read_exact_at(&mut head, 0).map_err(err)?;
        if &head[..8] != MAGIC && &head[..8] != MAGIC_V3 {
            return Err(format!("{}: not a packed pawnless table", path.display()));
        }
        let len = u64::from_le_bytes(head[8..].try_into().expect("8 bytes"));
        if len > 1 << 20 {
            return Err("table header too large".into());
        }
        let mut json = vec![0u8; len as usize];
        file.read_exact_at(&mut json, 16).map_err(err)?;
        let header: PackedHeader = serde_json::from_slice(&json).map_err(|e| e.to_string())?;
        let material = Material::parse(&header.material)?;
        let layout = Layout::new(&material);
        if header.stored_words != layout.words() {
            return Err("v3 header disagrees with its material".into());
        }
        if !header.block_bytes.is_power_of_two()
            || !(1 << 12..=1 << 24).contains(&header.block_bytes)
        {
            return Err("v3 block size out of range".into());
        }
        let start = 16 + len;
        let size = file.metadata().map_err(err)?.len();
        if authenticate {
            let mut payload = vec![0u8; (size - start) as usize];
            file.read_exact_at(&mut payload, start).map_err(err)?;
            if payload_digest(&payload, &header.digest_scheme)? != header.payload_sha256 {
                return Err(format!("{}: payload digest mismatch", path.display()));
            }
        }
        let index = |s: &Section| -> Result<SectionIndex> {
            let mut raw = vec![0u8; 8 * (s.blocks as usize + 1)];
            file.read_exact_at(&mut raw, start + s.offset)
                .map_err(err)?;
            let offsets: Vec<u64> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect();
            let frames = start + s.offset + raw.len() as u64;
            if offsets.windows(2).any(|w| w[0] > w[1]) || frames + offsets[offsets.len() - 1] > size
            {
                return Err("v3 block index out of range".into());
            }
            Ok(SectionIndex {
                frames,
                offsets,
                raw_bytes: s.raw_bytes as usize,
                codec: s.codec,
            })
        };
        let find = |name: &str| header.sections.iter().find(|s| s.name == name);
        let wdl = index(find("wdl").ok_or("v3 table lacks a wdl section")?)?;
        let rank = find("rank").map(index).transpose()?;
        if (rank.is_some()) != (header.rank_width > 0) {
            return Err("v3 rank section disagrees with the rank width".into());
        }
        Ok(PackedTable {
            header,
            layout,
            geometry: Geometry::new(),
            file,
            wdl,
            rank,
            cache: Mutex::new(LruCache::new(NonZeroUsize::new(BLOCK_CACHE).expect("> 0"))),
        })
    }

    pub fn material(&self) -> &Material {
        &self.layout.material
    }

    fn read_block(&self, section: &SectionIndex, b: usize) -> Result<Vec<u8>> {
        let (from, to) = (section.offsets[b], section.offsets[b + 1]);
        let mut frame = vec![0u8; (to - from) as usize];
        self.file
            .read_exact_at(&mut frame, section.frames + from)
            .map_err(|e| e.to_string())?;
        let block = self.header.block_bytes;
        let expect = block.min(section.raw_bytes - b * block);
        let raw = decompress_block(section.codec, &frame, expect)?;
        if raw.len() != expect {
            return Err("v3 block has the wrong length".into());
        }
        Ok(raw)
    }

    fn block(&self, rank: bool, b: usize) -> Result<Arc<Vec<u8>>> {
        if let Some(hit) = self.cache.lock().expect("cache").get(&(rank, b)) {
            return Ok(hit.clone());
        }
        let section = if rank {
            self.rank.as_ref().ok_or("no rank section")?
        } else {
            &self.wdl
        };
        let data = Arc::new(self.read_block(section, b)?);
        self.cache
            .lock()
            .expect("cache")
            .put((rank, b), data.clone());
        Ok(data)
    }

    /// Value of a position given in the full layout, or None if illegal.
    /// Identical pieces may be listed in any order.
    pub fn probe(&self, black: bool, squares: &[u8], black_king: u8) -> Result<Option<Value>> {
        let material = self.material();
        if squares.len() != material.skeleton_len() || black_king >= 64 {
            return Err("position does not fit the material".into());
        }
        let word = material.encode(black, squares);
        if statics(&self.geometry, material, word).valid >> black_king & 1 == 0 {
            return Ok(None);
        }
        let p = self.layout.position(black, squares, black_king);
        let code = self.code(p)?;
        let derived = black && self.header.rank_sides == RankSides::White;
        let wdl = match (code, derived) {
            (0, _) => 0,
            (1, _) | (3, true) => 2,
            (2, _) => -2,
            _ => return Err("invalid WDL code".into()),
        };
        let rank = match (wdl, self.header.rank_width, derived) {
            (0, _, _) | (_, 0, _) => None,
            (_, _, false) => Some(self.stored_rank(p)?),
            _ if code == 3 => Some(1),
            _ => {
                let lower = lower_materials(material);
                let lanes = 1u64 << black_king;
                let win = if wdl == 2 { lanes } else { 0 };
                let ranks = derive_black(&self.geometry, material, &lower, self, word, lanes, win)?;
                Some(ranks[usize::from(black_king)])
            }
        };
        Ok(Some(Value { wdl, rank }))
    }

    /// WDL code of stored position `p`.
    fn code(&self, p: usize) -> Result<u8> {
        let block = self.header.block_bytes;
        Ok(self.block(false, p / 4 / block)?[p / 4 % block] >> (2 * (p % 4)) & 3)
    }

    /// Stored proof rank of stored position `p` (a stored side to move).
    fn stored_rank(&self, p: usize) -> Result<u16> {
        let width = self.header.rank_width;
        let block = self.header.block_bytes;
        let at = p * width;
        let data = self.block(true, at / block)?;
        let i = at % block;
        Ok(if width == 1 {
            u16::from(data[i])
        } else {
            u16::from_le_bytes([data[i], data[i + 1]])
        })
    }

    fn section_bytes(&self, section: &SectionIndex, threads: usize) -> Result<Vec<u8>> {
        let blocks = parallel(section.offsets.len() - 1, threads, |b| {
            self.read_block(section, b)
        })?;
        Ok(blocks.concat())
    }

    /// Decode into the full working layout (every symmetric image).
    pub fn decode(&self, threads: usize) -> Result<Table> {
        let layout = &self.layout;
        let material = layout.material.clone();
        let wdl = self.section_bytes(&self.wdl, threads)?;
        let width = self.header.rank_width;
        let rank_raw = match &self.rank {
            Some(s) => self.section_bytes(s, threads)?,
            None => Vec::new(),
        };
        let ranked = width > 0;
        let stored_words = layout.words();
        let derived = self.header.rank_sides == RankSides::White;
        let rank_words = if derived {
            black_start(layout)
        } else {
            stored_words
        };
        let sc = 1usize << 12;
        let stored_parts = parallel(stored_words.div_ceil(sc), threads, |part| {
            let range = part * sc..((part + 1) * sc).min(stored_words);
            let mut planes = Vec::with_capacity(range.len());
            let mut ranks = Vec::with_capacity(if ranked { range.len() * 64 } else { 0 });
            for cw in range {
                let w = layout.full_word(cw);
                let valid = statics(&self.geometry, &material, w).valid;
                let (mut win, mut loss) = (0u64, 0u64);
                let mut word_rank = [UNRESOLVED; 64];
                for (bk, slot) in word_rank.iter_mut().enumerate() {
                    let p = cw * 64 + bk;
                    let code = wdl[p / 4] >> (2 * (p % 4)) & 3;
                    if valid >> bk & 1 != 0 {
                        win |= u64::from(code & 1 == 1) << bk;
                        loss |= u64::from(code == 2) << bk;
                        if code == 3 {
                            *slot = 1;
                        } else if ranked && code != 0 && cw < rank_words {
                            *slot = if width == 1 {
                                u16::from(rank_raw[p])
                            } else {
                                u16::from_le_bytes([rank_raw[2 * p], rank_raw[2 * p + 1]])
                            };
                        }
                    }
                }
                planes.push([valid, win, loss]);
                if ranked {
                    ranks.extend_from_slice(&word_rank);
                }
            }
            Ok((planes, ranks))
        })?;
        let mut s_planes = Vec::with_capacity(stored_words);
        let mut s_ranks = Vec::with_capacity(if ranked { stored_words * 64 } else { 0 });
        for (p, r) in stored_parts {
            s_planes.extend_from_slice(&p);
            s_ranks.extend_from_slice(&r);
        }

        let words = material.words();
        let mut table = Table {
            material: material.clone(),
            valid: vec![0u64; words],
            win: vec![0u64; words],
            loss: vec![0u64; words],
            rank: vec![0u16; if ranked { words * 64 } else { 0 }],
            rounds: self.header.rounds,
        };
        // Every other word is a symmetric image of a stored word: map its
        // planes with O(1) bitboard symmetries instead of re-deriving them.
        let chunk = 1usize << 16;
        let mut rank_chunks = table.rank.chunks_mut(chunk * 64);
        let work: Vec<_> = table
            .valid
            .chunks_mut(chunk)
            .zip(table.win.chunks_mut(chunk))
            .zip(table.loss.chunks_mut(chunk))
            .enumerate()
            .map(|(part, ((v, w), l))| (part * chunk, v, w, l, rank_chunks.next()))
            .collect();
        let workers = threads.clamp(1, work.len().max(1));
        let work = Mutex::new(work.into_iter());
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let next = work.lock().expect("work").next();
                        let Some((w0, vs, ws, ls, mut rs)) = next else {
                            break;
                        };
                        for i in 0..vs.len() {
                            let (black, squares) = material.decode(w0 + i);
                            let image = layout.image(black, &squares[..layout.m]);
                            let [sv, sw, sl] = s_planes[image.word];
                            vs[i] = image.bitboard(sv);
                            ws[i] = image.bitboard(sw);
                            ls[i] = image.bitboard(sl);
                            if let Some(r) = rs.as_deref_mut() {
                                let dst = &mut r[i * 64..(i + 1) * 64];
                                dst.fill(UNRESOLVED);
                                let mut bits = ws[i] | ls[i];
                                while bits != 0 {
                                    let bk = bits.trailing_zeros() as u8;
                                    bits &= bits - 1;
                                    dst[usize::from(bk)] =
                                        s_ranks[image.word * 64 + usize::from(image.square(bk))];
                                }
                            }
                        }
                    }
                });
            }
        });
        if ranked && derived {
            self.derive_black_ranks(&mut table, threads)?;
        }
        Ok(table)
    }

    /// Fill the black-to-move ranks that a white-only file leaves unresolved.
    fn derive_black_ranks(&self, table: &mut Table, threads: usize) -> Result<()> {
        let material = &table.material;
        let lower = lower_materials(material);
        let half = material.words() / 2;
        let (white, black) = table.rank.split_at_mut(half * 64);
        let kids = Decoded {
            valid: &table.valid,
            loss: &table.loss,
            rank: white,
        };
        let chunk = 1usize << 12;
        let work = Mutex::new(black.chunks_mut(chunk * 64).enumerate());
        let failure = Mutex::new(None);
        std::thread::scope(|scope| {
            for _ in 0..threads.max(1) {
                scope.spawn(|| {
                    loop {
                        let next = work.lock().expect("work").next();
                        let Some((part, ranks)) = next else { break };
                        for (i, dst) in ranks.chunks_mut(64).enumerate() {
                            let w = half + part * chunk + i;
                            let win = table.win[w];
                            let mut lanes = table.loss[w];
                            let mut wins = win;
                            while wins != 0 {
                                let bk = wins.trailing_zeros() as usize;
                                wins &= wins - 1;
                                if dst[bk] != 1 {
                                    lanes |= 1 << bk;
                                }
                            }
                            let derived = derive_black(
                                &self.geometry,
                                material,
                                &lower,
                                &kids,
                                w,
                                lanes,
                                win,
                            );
                            match derived {
                                Ok(r) => {
                                    let mut bits = lanes;
                                    while bits != 0 {
                                        let bk = bits.trailing_zeros() as usize;
                                        bits &= bits - 1;
                                        dst[bk] = r[bk];
                                    }
                                }
                                Err(e) => {
                                    *failure.lock().expect("failure") = Some(e);
                                    return;
                                }
                            }
                        }
                    }
                });
            }
        });
        match failure.into_inner().expect("failure") {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Read a table file of any version (v1, v2 or v3) into the working layout;
/// returns the table and a payload digest identifying the file.
pub fn load_table(path: &Path, threads: usize) -> Result<(Table, String)> {
    if is_packed(path)? {
        let packed = PackedTable::open(path, true)?;
        let table = packed.decode(threads)?;
        Ok((table, packed.header.payload_sha256))
    } else {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let (table, header) = Table::read(std::io::BufReader::new(file))?;
        Ok((table, header.payload_sha256))
    }
}
