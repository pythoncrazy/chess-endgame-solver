//! GPU pawnless retrograde solver (cuTile). Same layout and synchronous rounds
//! as `tablebase_core::pawnless`. The CPU reference is only for comparison.
//!
//! One tile lane handles one word: the side to move and all skeleton squares.
//! The 64 bits of the word give the black-king square. Device state is one
//! record per word, `[win | illegal, loss | illegal]`. If both bits are set, the
//! position is illegal. Thus the value of a child is one read of a 16-byte pair.
//! A 64-word block shares the side to move. Thus a uniform branch skips the
//! pieces of the other side. Each role walks only its own directions and steps.
//! Synchronous rounds read only the previous round, so the round number
//! is a proof rank.
use cutile::jit_cache::{self, FileSystemJitStore};
use cutile::prelude::*;
use serde_json::json;
use std::{fs::File, io::BufWriter, path::PathBuf, sync::Arc, time::Instant};
use tablebase_core::packed::{self, Canonical, Layout, PackedTable};
use tablebase_core::pawnless::{self, Counts, Material, Solved, Table, UNRESOLVED, solve_tree};

/// Program and git revision recorded in table headers (see build.rs).
const GENERATOR: &str = concat!("gpu_solve git:", env!("TB_SOURCE_REVISION"));
/// Host threads that convert downloaded records. The v3 writer runs beside them.
const CONVERT_THREADS: usize = 8;
/// Lanes (words) per tile block.
const BLOCK: usize = 64;
/// i32 metadata: m is at 0. Piece i is at 8 + 4i (role, white, directions, slides).
/// Direction d of role r is at 64 + 2(8r + d) (file step, rank step).
const META_LEN: usize = 64 + 5 * 8 * 2;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    /// Bit-parallel king neighbourhood.
    fn king_reach(x: Tile<u64, { [64] }>) -> Tile<u64, { [64] }> {
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let eight: Tile<u64, { [64] }> = constant(8u64, shape![64]);
        let not_h: Tile<u64, { [64] }> = constant(0x7f7f7f7f7f7f7f7fu64, shape![64]);
        let not_a: Tile<u64, { [64] }> = not_h << one;
        let sideways: Tile<u64, { [64] }> = ((x & not_h) << one) | ((x & not_a) >> one);
        let row: Tile<u64, { [64] }> = x | sideways;
        sideways | (row << eight) | (row >> eight)
    }

    /// Lane mask (all ones or zero) from a predicate.
    fn lanes(condition: Tile<bool, { [64] }>) -> Tile<u64, { [64] }> {
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let ones: Tile<u64, { [64] }> = zero - one;
        select(condition, ones, zero)
    }

    /// `1 << square` as a scalar bitboard.
    fn square_bit(square: i32) -> u64 {
        let s: Tile<i32, { [] }> = scalar_to_tile(square);
        let s: Tile<i64, { [] }> = exti(s);
        let s: Tile<u64, { [] }> = bitcast(s);
        let one: Tile<u64, { [] }> = scalar_to_tile(1u64);
        tile_to_scalar(one << s)
    }

    fn i64s(value: i32) -> Tile<i64, { [64] }> {
        let t: Tile<i32, { [64] }> = broadcast_scalar(value, shape![64]);
        exti(t)
    }

    /// Scalar block index of the 2-D grid. Block ids above 2^24 on one axis
    /// are unreliable in the pinned cuTile. Thus large launches use grid Y.
    fn block_id() -> i32 {
        let pid = get_tile_block_id();
        let grid = get_num_tile_blocks();
        pid.1 * grid.0 + pid.0
    }

    fn word_index(block: i32) -> Tile<i64, { [64] }> {
        let width: Tile<i64, { [64] }> = constant(64i64, shape![64]);
        let lane: Tile<i64, { [64] }> = iota(shape![64]);
        i64s(block) * width + lane
    }

    unsafe fn scalar_at(table: *const i32, index: i32) -> i32 {
        let base: PointerTile<*const i32, { [] }> = pointer_to_tile(table);
        let ptr: PointerTile<*const i32, { [] }> = base.offset(index);
        let loaded: (Tile<i32, { [] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        tile_to_scalar(loaded.0)
    }

    unsafe fn scalar_u16_at(table: *const u16, index: i32) -> u16 {
        let base: PointerTile<*const u16, { [] }> = pointer_to_tile(table);
        let ptr: PointerTile<*const u16, { [] }> = base.offset(index);
        let loaded: (Tile<u16, { [] }>, Token) = unsafe {
            load_ptr_tko(
                ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        tile_to_scalar(loaded.0)
    }

    /// Lane mask (all ones or zero) of the nonzero flags `table[index[lane]]`.
    unsafe fn load_lanes(table: *const i32, index: Tile<i64, { [64] }>) -> Tile<u64, { [64] }> {
        let base: PointerTile<*const i32, { [] }> = pointer_to_tile(table);
        let base: PointerTile<*const i32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const i32, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*const i32, { [64] }> = base.offset_tile(index);
        let loaded: (Tile<i32, { [64] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        let zero: Tile<i32, { [64] }> = constant(0i32, shape![64]);
        lanes(ne_tile(loaded.0, zero))
    }

    /// Store one or zero to `table[index[lane]]` as `mask` is nonzero or zero.
    unsafe fn store_lanes(table: *mut i32, index: Tile<i64, { [64] }>, mask: Tile<u64, { [64] }>) {
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let one: Tile<i32, { [64] }> = constant(1i32, shape![64]);
        let none: Tile<i32, { [64] }> = constant(0i32, shape![64]);
        let value: Tile<i32, { [64] }> = select(ne_tile(mask, zero), one, none);
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(table);
        let base: PointerTile<*mut i32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut i32, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*mut i32, { [64] }> = base.offset_tile(index);
        unsafe {
            store_ptr_tko(
                ptrs,
                value,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            )
        };
    }

    /// Number of lanes whose mask is nonzero.
    fn count_lanes(mask: Tile<u64, { [64] }>) -> i64 {
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let one_i: Tile<i64, { [64] }> = constant(1i64, shape![64]);
        let zero_i: Tile<i64, { [64] }> = constant(0i64, shape![64]);
        let n: Tile<i64, { [] }> = reduce_sum(select(ne_tile(mask, zero), one_i, zero_i), 0);
        tile_to_scalar(n)
    }

    /// Set flag `index[lane]` to one for lanes where `mask` is nonzero.
    unsafe fn mark_flags(table: *mut i32, index: Tile<i64, { [64] }>, mask: Tile<u64, { [64] }>) {
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let keep: Tile<bool, { [64] }> = ne_tile(mask, zero);
        let base: PointerTile<*mut i32, { [] }> = pointer_to_tile(table);
        let base: PointerTile<*mut i32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut i32, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*mut i32, { [64] }> = base.offset_tile(index);
        let ones: Tile<i32, { [64] }> = constant(1i32, shape![64]);
        unsafe {
            store_ptr_tko(
                ptrs,
                ones,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(keep),
                None,
                Latency::<0>,
            )
        };
    }

    /// Masked gather of records `[win | illegal, loss | illegal]`. Masked lanes
    /// read zero.
    unsafe fn gather_pair(
        records: *const u64,
        index: Tile<i64, { [64] }>,
        mask: Tile<u64, { [64] }>,
    ) -> Tile<u64, { [64, 2] }> {
        let two: Tile<i64, { [64] }> = constant(2i64, shape![64]);
        let first: Tile<i64, { [64] }> = index * two;
        let first: Tile<i64, { [64, 2] }> = first.reshape(shape![64, 1]).broadcast(shape![64, 2]);
        let col: Tile<i64, { [2] }> = iota(shape![2]);
        let col: Tile<i64, { [64, 2] }> = col.reshape(shape![1, 2]).broadcast(shape![64, 2]);
        // Masked 2-D loads do not lower in the pinned cuTile, and bool tiles
        // cannot be broadcast. Thus masked-off lanes read record 0, which is
        // always valid. Then the widened u64 lane mask sets them to zero.
        let mask: Tile<u64, { [64, 2] }> = mask.reshape(shape![64, 1]).broadcast(shape![64, 2]);
        let zero2: Tile<u64, { [64, 2] }> = constant(0u64, shape![64, 2]);
        let keep: Tile<bool, { [64, 2] }> = ne_tile(mask, zero2);
        let idx: Tile<i64, { [64, 2] }> = select(keep, first + col, col);
        let base: PointerTile<*const u64, { [] }> = pointer_to_tile(records);
        let base: PointerTile<*const u64, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*const u64, { [64, 2] }> = base.broadcast(shape![64, 2]);
        let ptrs: PointerTile<*const u64, { [64, 2] }> = base.offset_tile(idx);
        let loaded: (Tile<u64, { [64, 2] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        loaded.0 & mask
    }

    fn pair_part(pair: Tile<u64, { [64, 2] }>, part: i64) -> Tile<u64, { [64] }> {
        let col: Tile<i64, { [2] }> = iota(shape![2]);
        let col: Tile<i64, { [64, 2] }> = col.reshape(shape![1, 2]).broadcast(shape![64, 2]);
        let want: Tile<i64, { [64, 2] }> = broadcast_scalar(part, shape![64, 2]);
        let zero: Tile<u64, { [64, 2] }> = constant(0u64, shape![64, 2]);
        reduce_sum(select(eq_tile(col, want), pair, zero), 1)
    }

    unsafe fn store_pair(
        records: *mut u64,
        index: Tile<i64, { [64] }>,
        win: Tile<u64, { [64] }>,
        loss: Tile<u64, { [64] }>,
    ) {
        let two: Tile<i64, { [64] }> = constant(2i64, shape![64]);
        let first: Tile<i64, { [64] }> = index * two;
        let first: Tile<i64, { [64, 2] }> = first.reshape(shape![64, 1]).broadcast(shape![64, 2]);
        let col: Tile<i64, { [2] }> = iota(shape![2]);
        let col: Tile<i64, { [64, 2] }> = col.reshape(shape![1, 2]).broadcast(shape![64, 2]);
        let zero: Tile<i64, { [64, 2] }> = constant(0i64, shape![64, 2]);
        let w2: Tile<u64, { [64, 2] }> = win.reshape(shape![64, 1]).broadcast(shape![64, 2]);
        let l2: Tile<u64, { [64, 2] }> = loss.reshape(shape![64, 1]).broadcast(shape![64, 2]);
        let value: Tile<u64, { [64, 2] }> = select(eq_tile(col, zero), w2, l2);
        let base: PointerTile<*mut u64, { [] }> = pointer_to_tile(records);
        let base: PointerTile<*mut u64, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*mut u64, { [64, 2] }> = base.broadcast(shape![64, 2]);
        let ptrs: PointerTile<*mut u64, { [64, 2] }> = base.offset_tile(first + col);
        unsafe {
            store_ptr_tko(
                ptrs,
                value,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            )
        };
    }

    unsafe fn gather(plane: *const u64, index: Tile<i64, { [64] }>) -> Tile<u64, { [64] }> {
        let base: PointerTile<*const u64, { [] }> = pointer_to_tile(plane);
        let base: PointerTile<*const u64, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const u64, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*const u64, { [64] }> = base.offset_tile(index);
        let loaded: (Tile<u64, { [64] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            )
        };
        loaded.0
    }

    unsafe fn scatter(plane: *mut u64, index: Tile<i64, { [64] }>, value: Tile<u64, { [64] }>) {
        let base: PointerTile<*mut u64, { [] }> = pointer_to_tile(plane);
        let base: PointerTile<*mut u64, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut u64, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*mut u64, { [64] }> = base.offset_tile(index);
        unsafe {
            store_ptr_tko(
                ptrs,
                value,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            )
        };
    }

    // Half-split tiles: element 2 * lane + half, half 0 = black king squares 0..32.
    fn col_index() -> Tile<i32, { [128] }> {
        let e: Tile<i32, { [128] }> = iota(shape![128]);
        e & constant(1i32, shape![128])
    }

    fn row_index() -> Tile<i32, { [128] }> {
        let e: Tile<i32, { [128] }> = iota(shape![128]);
        e >> constant(1i32, shape![128])
    }

    fn i32h(value: i32) -> Tile<i32, { [128] }> {
        broadcast_scalar(value, shape![128])
    }

    fn i64h(value: i32) -> Tile<i64, { [128] }> {
        exti(i32h(value))
    }

    fn lanes_h(condition: Tile<bool, { [128] }>) -> Tile<u32, { [128] }> {
        let zero: Tile<u32, { [128] }> = constant(0u32, shape![128]);
        let one: Tile<u32, { [128] }> = constant(1u32, shape![128]);
        select(condition, zero - one, zero)
    }

    /// `1 << (square & 31)` as a u32 scalar.
    fn bit32(square: i32) -> u32 {
        let s: Tile<i32, { [] }> = scalar_to_tile(square & 31i32);
        let s: Tile<u32, { [] }> = bitcast(s);
        let one: Tile<u32, { [] }> = scalar_to_tile(1u32);
        tile_to_scalar(one << s)
    }

    /// Low and high halves of `1 << square` (zero off the board).
    fn lo_bit(square: i32) -> u32 {
        if square < 32i32 { bit32(square) } else { 0u32 }
    }

    fn hi_bit(square: i32) -> u32 {
        let mut b: u32 = 0u32;
        if square >= 32i32 {
            if square < 64i32 {
                b = bit32(square);
            }
        }
        b
    }

    /// Per element, the half `col` of a 64-bit scalar given as halves.
    fn pick(col: Tile<i32, { [128] }>, lo: u32, hi: u32) -> Tile<u32, { [128] }> {
        let zero: Tile<i32, { [128] }> = constant(0i32, shape![128]);
        let lo_t: Tile<u32, { [128] }> = broadcast_scalar(lo, shape![128]);
        let hi_t: Tile<u32, { [128] }> = broadcast_scalar(hi, shape![128]);
        select(eq_tile(col, zero), lo_t, hi_t)
    }

    /// All ones where bit `square` (0..64) of the 64-bit value (lo, hi) is set.
    fn test_bit(lo: u32, hi: u32, square: Tile<i32, { [128] }>) -> Tile<u32, { [128] }> {
        let thirty_two: Tile<i32, { [128] }> = constant(32i32, shape![128]);
        let thirty_one: Tile<i32, { [128] }> = constant(31i32, shape![128]);
        let lo_t: Tile<u32, { [128] }> = broadcast_scalar(lo, shape![128]);
        let hi_t: Tile<u32, { [128] }> = broadcast_scalar(hi, shape![128]);
        let v: Tile<u32, { [128] }> = select(lt_tile(square, thirty_two), lo_t, hi_t);
        let sh: Tile<u32, { [128] }> = bitcast(square & thirty_one);
        let one: Tile<u32, { [128] }> = constant(1u32, shape![128]);
        let zero: Tile<u32, { [128] }> = constant(0u32, shape![128]);
        lanes_h(ne_tile((v >> sh) & one, zero))
    }

    /// Half `col` of the bitboard `1 << square` (square on the board).
    fn half_bit(square: Tile<i32, { [128] }>, col: Tile<i32, { [128] }>) -> Tile<u32, { [128] }> {
        let five: Tile<i32, { [128] }> = constant(5i32, shape![128]);
        let thirty_one: Tile<i32, { [128] }> = constant(31i32, shape![128]);
        let one: Tile<u32, { [128] }> = constant(1u32, shape![128]);
        let zero: Tile<u32, { [128] }> = constant(0u32, shape![128]);
        let sh: Tile<u32, { [128] }> = bitcast(square & thirty_one);
        select(eq_tile(square >> five, col), one << sh, zero)
    }

    /// Both u32 halves of each lane, low half first.
    fn halves(x: Tile<u64, { [64] }>) -> Tile<u32, { [128] }> {
        let x2: Tile<u64, { [64, 2] }> = x.reshape(shape![64, 1]).broadcast(shape![64, 2]);
        let c: Tile<u64, { [2] }> = iota(shape![2]);
        let c: Tile<u64, { [64, 2] }> = c.reshape(shape![1, 2]).broadcast(shape![64, 2]);
        let thirty_two: Tile<u64, { [64, 2] }> = constant(32u64, shape![64, 2]);
        let low: Tile<u64, { [64, 2] }> = constant(0xffff_ffffu64, shape![64, 2]);
        let h: Tile<u32, { [64, 2] }> =
            trunci((x2 >> (c * thirty_two)) & low, overflow::NoUnsignedWrap);
        h.reshape(shape![128])
    }

    fn join(h: Tile<u32, { [128] }>) -> Tile<u64, { [64] }> {
        let h: Tile<u32, { [64, 2] }> = h.reshape(shape![64, 2]);
        let wide: Tile<u64, { [64, 2] }> = exti(h);
        let c: Tile<u64, { [2] }> = iota(shape![2]);
        let c: Tile<u64, { [64, 2] }> = c.reshape(shape![1, 2]).broadcast(shape![64, 2]);
        let thirty_two: Tile<u64, { [64, 2] }> = constant(32u64, shape![64, 2]);
        reduce_sum(wide << (c * thirty_two), 1)
    }

    fn count_h(mask: Tile<u32, { [128] }>) -> i64 {
        let zero: Tile<u32, { [128] }> = constant(0u32, shape![128]);
        let one_i: Tile<i64, { [128] }> = constant(1i64, shape![128]);
        let zero_i: Tile<i64, { [128] }> = constant(0i64, shape![128]);
        let n: Tile<i64, { [] }> = reduce_sum(select(ne_tile(mask, zero), one_i, zero_i), 0);
        tile_to_scalar(n)
    }

    /// Record of the lower table after a capture of skeleton piece `j` on the
    /// move to `moved`. This is table j - 1, without piece j, with the other side to move.
    fn lower_child(
        moved: Tile<i64, { [128] }>,
        j: Tile<i64, { [128] }>,
        m: Tile<i64, { [128] }>,
        lower_words: Tile<i64, { [128] }>,
        lower_stm: Tile<i64, { [128] }>,
    ) -> Tile<i64, { [128] }> {
        let one: Tile<i64, { [128] }> = constant(1i64, shape![128]);
        let six: Tile<i64, { [128] }> = constant(6i64, shape![128]);
        let low_shift: Tile<i64, { [128] }> = six * (m - one - j);
        let low_mask: Tile<i64, { [128] }> = (one << low_shift) - one;
        let hi: Tile<i64, { [128] }> = moved >> (low_shift + six);
        let low_word: Tile<i64, { [128] }> = ((hi << low_shift) | (moved & low_mask)) ^ lower_stm;
        (j - one) * lower_words + low_word
    }

    /// Masked gather of half `col` of record word `part` (0 win, 2 loss) of
    /// `records` as u32. Masked-off lanes give zero.
    unsafe fn gather_half(
        records: *const u32,
        record: Tile<i64, { [128] }>,
        col: Tile<i64, { [128] }>,
        mask: Tile<u32, { [128] }>,
        part: i64,
    ) -> Tile<u32, { [128] }> {
        let zero: Tile<u32, { [128] }> = constant(0u32, shape![128]);
        let four: Tile<i64, { [128] }> = constant(4i64, shape![128]);
        let p: Tile<i64, { [128] }> = broadcast_scalar(part, shape![128]);
        let base: PointerTile<*const u32, { [] }> = pointer_to_tile(records);
        let base: PointerTile<*const u32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const u32, { [128] }> = base.broadcast(shape![128]);
        let ptrs: PointerTile<*const u32, { [128] }> = base.offset_tile(record * four + p + col);
        let loaded: (Tile<u32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(ne_tile(mask, zero)),
                Some(0u32),
                None,
                Latency::<0>,
            )
        };
        loaded.0
    }

    /// Masked gather of `records` (u32 view) at block `block` (64 records,
    /// 256 u32) plus per-element `offsets`. Masked-off lanes give zero.
    unsafe fn gather_block(
        records: *const u32,
        block: i32,
        offsets: Tile<i32, { [128] }>,
        mask: Tile<u32, { [128] }>,
    ) -> Tile<u32, { [128] }> {
        let zero: Tile<u32, { [128] }> = constant(0u32, shape![128]);
        let b: Tile<i32, { [] }> = scalar_to_tile(block);
        let b: Tile<i64, { [] }> = exti(b);
        let per_block: Tile<i64, { [] }> = scalar_to_tile(256i64);
        let base: PointerTile<*const u32, { [] }> = pointer_to_tile(records);
        let base: PointerTile<*const u32, { [] }> = base.offset_tile(b * per_block);
        let base: PointerTile<*const u32, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const u32, { [128] }> = base.broadcast(shape![128]);
        let ptrs: PointerTile<*const u32, { [128] }> = base.offset_tile(offsets);
        let loaded: (Tile<u32, { [128] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(ne_tile(mask, zero)),
                Some(0u32),
                None,
                Latency::<0>,
            )
        };
        loaded.0
    }

    /// Block of the lower table without skeleton piece `j` (1..m-1), with the
    /// other side to move. Block `block` of an m-piece table maps to it.
    fn lower_block(block: i32, j: i32, m: i32) -> i32 {
        let low_shift: i32 = 6i32 * (m - 2i32 - j);
        let hi: i32 = block >> (low_shift + 6i32);
        let low: i32 = block & ((1i32 << low_shift) - 1i32);
        (((hi << low_shift) | low) ^ (1i32 << (6i32 * m - 12i32)))
            + (j - 1i32) * (1i32 << (6i32 * m - 11i32))
    }

    fn sym_square(sym: i32, mut s: Tile<i64, { [64] }>) -> Tile<i64, { [64] }> {
        let three: Tile<i64, { [64] }> = constant(3i64, shape![64]);
        let seven: Tile<i64, { [64] }> = constant(7i64, shape![64]);
        let fifty_six: Tile<i64, { [64] }> = constant(56i64, shape![64]);
        if (sym & 1i32) != 0i32 {
            s = ((s & seven) << three) | (s >> three);
        }
        if (sym & 2i32) != 0i32 {
            s = s ^ seven;
        }
        if (sym & 4i32) != 0i32 {
            s = s ^ fifty_six;
        }
        s
    }

    fn sym_bitboard(sym: i32, mut x: Tile<u64, { [64] }>) -> Tile<u64, { [64] }> {
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let two: Tile<u64, { [64] }> = constant(2u64, shape![64]);
        let four: Tile<u64, { [64] }> = constant(4u64, shape![64]);
        let seven: Tile<u64, { [64] }> = constant(7u64, shape![64]);
        let eight: Tile<u64, { [64] }> = constant(8u64, shape![64]);
        let fourteen: Tile<u64, { [64] }> = constant(14u64, shape![64]);
        let sixteen: Tile<u64, { [64] }> = constant(16u64, shape![64]);
        let twenty_eight: Tile<u64, { [64] }> = constant(28u64, shape![64]);
        let thirty_two: Tile<u64, { [64] }> = constant(32u64, shape![64]);
        if (sym & 1i32) != 0i32 {
            let m28: Tile<u64, { [64] }> = constant(0x0f0f0f0f00000000u64, shape![64]);
            let t: Tile<u64, { [64] }> = m28 & (x ^ (x << twenty_eight));
            x = x ^ t ^ (t >> twenty_eight);
            let m14: Tile<u64, { [64] }> = constant(0x3333000033330000u64, shape![64]);
            let t: Tile<u64, { [64] }> = m14 & (x ^ (x << fourteen));
            x = x ^ t ^ (t >> fourteen);
            let m7: Tile<u64, { [64] }> = constant(0x5500550055005500u64, shape![64]);
            let t: Tile<u64, { [64] }> = m7 & (x ^ (x << seven));
            x = x ^ t ^ (t >> seven);
        }
        if (sym & 2i32) != 0i32 {
            let m1: Tile<u64, { [64] }> = constant(0x5555555555555555u64, shape![64]);
            let m2: Tile<u64, { [64] }> = constant(0x3333333333333333u64, shape![64]);
            let m4: Tile<u64, { [64] }> = constant(0x0f0f0f0f0f0f0f0fu64, shape![64]);
            x = ((x >> one) & m1) | ((x & m1) << one);
            x = ((x >> two) & m2) | ((x & m2) << two);
            x = ((x >> four) & m4) | ((x & m4) << four);
        }
        if (sym & 4i32) != 0i32 {
            // Byte swap from masks below 2^63 (larger u64 constants lower
            // incorrectly in the pinned cuTile).
            let b1: Tile<u64, { [64] }> = constant(0x00ff00ff00ff00ffu64, shape![64]);
            let b2: Tile<u64, { [64] }> = constant(0x0000ffff0000ffffu64, shape![64]);
            let b4: Tile<u64, { [64] }> = constant(0x00000000ffffffffu64, shape![64]);
            x = ((x >> eight) & b1) | ((x & b1) << eight);
            x = ((x >> sixteen) & b2) | ((x & b2) << sixteen);
            x = (x >> thirty_two) | ((x & b4) << thirty_two);
        }
        x
    }

    /// Set bits over the block's 64 words.
    fn popcount_sum(x: Tile<u64, { [64] }>) -> Tile<i64, { [] }> {
        let col: Tile<u64, { [64] }> = iota(shape![64]);
        let col: Tile<u64, { [64, 64] }> = col.reshape(shape![1, 64]).broadcast(shape![64, 64]);
        let x2: Tile<u64, { [64, 64] }> = x.reshape(shape![64, 1]).broadcast(shape![64, 64]);
        let one2: Tile<u64, { [64, 64] }> = constant(1u64, shape![64, 64]);
        let zero2: Tile<u64, { [64, 64] }> = constant(0u64, shape![64, 64]);
        let one_i2: Tile<i64, { [64, 64] }> = constant(1i64, shape![64, 64]);
        let zero_i2: Tile<i64, { [64, 64] }> = constant(0i64, shape![64, 64]);
        let bits: Tile<i64, { [64, 64] }> =
            select(ne_tile((x2 >> col) & one2, zero2), one_i2, zero_i2);
        let rows: Tile<i64, { [64] }> = reduce_sum(bits, 1);
        reduce_sum(rows, 0)
    }

    unsafe fn atomic_add_at(table: *mut i64, index: i32, value: Tile<i64, { [] }>) {
        let value: Tile<i64, { [1] }> = value.reshape(shape![1]);
        let base: PointerTile<*mut i64, { [] }> = pointer_to_tile(table);
        let ptr: PointerTile<*mut i64, { [] }> = base.offset(index);
        let ptr: PointerTile<*mut i64, { [1] }> = ptr.reshape(shape![1]);
        let _old: (Tile<i64, { [1] }>, Token) = unsafe {
            atomic_rmw_tko(
                ptr,
                value,
                atomic::Add,
                ordering::Relaxed,
                scope::Device,
                None,
                None,
            )
        };
    }

    /// Final-state checks. `out[0]` counts words whose records differ from
    /// any of their seven symmetric images. `out[1 + 3 * turn + k]` counts
    /// legal (k = 0), won (1) and lost (2) positions per side to move.
    #[cutile::entry()]
    unsafe fn finish(out: *mut i64, records: *const u64, meta: *const i32) {
        let m: i32 = unsafe { scalar_at(meta, 0i32) };
        let blk: i32 = block_id();
        let w: Tile<i64, { [64] }> = word_index(blk);
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let ones: Tile<u64, { [64] }> = zero - one;
        let zero_i: Tile<i64, { [64] }> = constant(0i64, shape![64]);
        let one_i: Tile<i64, { [64] }> = constant(1i64, shape![64]);
        let sixty_three: Tile<i64, { [64] }> = constant(63i64, shape![64]);
        let black_to_move: i32 = (blk >> (6i32 * m - 6i32)) & 1i32;
        let stm_base: Tile<i64, { [64] }> = i64s(black_to_move) << i64s(6i32 * m);
        let op: Tile<u64, { [64, 2] }> = unsafe { gather_pair(records, w, ones) };
        let rw: Tile<u64, { [64] }> = pair_part(op, 0i64);
        let rl: Tile<u64, { [64] }> = pair_part(op, 1i64);
        let mut bad: Tile<u64, { [64] }> = zero;
        for sym in 1i32..8i32 {
            let mut v: Tile<i64, { [64] }> = stm_base;
            for i in 0i32..m {
                let shift: Tile<i64, { [64] }> = i64s(6i32 * (m - 1i32 - i));
                let sq: Tile<i64, { [64] }> = (w >> shift) & sixty_three;
                v = v | (sym_square(sym, sq) << shift);
            }
            let vp: Tile<u64, { [64, 2] }> = unsafe { gather_pair(records, v, ones) };
            bad = bad
                | (sym_bitboard(sym, rw) ^ pair_part(vp, 0i64))
                | (sym_bitboard(sym, rl) ^ pair_part(vp, 1i64));
        }
        let violations: Tile<i64, { [] }> =
            reduce_sum(select(ne_tile(bad, zero), one_i, zero_i), 0);
        let n: i64 = tile_to_scalar(violations);
        if n > 0i64 {
            unsafe { atomic_add_at(out, 0i32, violations) };
        }
        let valid: Tile<u64, { [64] }> = (rw & rl) ^ ones;
        let at: i32 = 1i32 + 3i32 * black_to_move;
        unsafe { atomic_add_at(out, at, popcount_sum(valid)) };
        unsafe { atomic_add_at(out, at + 1i32, popcount_sum(rw & valid)) };
        unsafe { atomic_add_at(out, at + 2i32, popcount_sum(rl & valid)) };
    }

    /// The start records, where illegal positions are both win and loss. Also
    /// the positions in which the side to move is in check.
    #[cutile::entry()]
    unsafe fn statics(records: *mut u64, check_out: *mut u64, meta: *const i32) {
        // Scalars come from device memory. Scalar kernel arguments go into the
        // in-memory kernel key of the pinned cuTile and force front-end reruns.
        let m: i32 = unsafe { scalar_at(meta, 0i32) };
        let blk: i32 = block_id();
        let w: Tile<i64, { [64] }> = word_index(blk);
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let ones: Tile<u64, { [64] }> = zero - one;
        let zero_i: Tile<i64, { [64] }> = constant(0i64, shape![64]);
        let three_i: Tile<i64, { [64] }> = constant(3i64, shape![64]);
        let seven_i: Tile<i64, { [64] }> = constant(7i64, shape![64]);
        let eight_i: Tile<i64, { [64] }> = constant(8i64, shape![64]);
        let sixty_three: Tile<i64, { [64] }> = constant(63i64, shape![64]);
        let black_to_move: i32 = (blk >> (6i32 * m - 6i32)) & 1i32;
        let black: Tile<u64, { [64] }> = lanes(ne_tile(i64s(black_to_move), zero_i));
        let mut occ: Tile<u64, { [64] }> = zero;
        let mut dup: Tile<u64, { [64] }> = zero;
        for i in 0i32..m {
            let sq: Tile<i64, { [64] }> = (w >> i64s(6i32 * (m - 1i32 - i))) & sixty_three;
            let bit: Tile<u64, { [64] }> = one << bitcast(sq);
            dup = dup | (occ & bit);
            occ = occ | bit;
        }
        let wk: Tile<i64, { [64] }> = (w >> i64s(6i32 * (m - 1i32))) & sixty_three;
        let mut white_att: Tile<u64, { [64] }> = zero;
        let mut wk_safe: Tile<u64, { [64] }> = ones;
        for i in 0i32..m {
            let role: i32 = unsafe { scalar_at(meta, 8i32 + 4i32 * i) };
            let white: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * i) };
            let dirs: i32 = unsafe { scalar_at(meta, 10i32 + 4i32 * i) };
            let slides: i32 = unsafe { scalar_at(meta, 11i32 + 4i32 * i) };
            let white_l: Tile<u64, { [64] }> = lanes(ne_tile(i64s(white), zero_i));
            let sq: Tile<i64, { [64] }> = (w >> i64s(6i32 * (m - 1i32 - i))) & sixty_three;
            let file: Tile<i64, { [64] }> = sq & seven_i;
            let rank: Tile<i64, { [64] }> = sq >> three_i;
            for d in 0i32..dirs {
                let df: Tile<i64, { [64] }> =
                    i64s(unsafe { scalar_at(meta, 64i32 + 16i32 * role + 2i32 * d) });
                let dr: Tile<i64, { [64] }> =
                    i64s(unsafe { scalar_at(meta, 65i32 + 16i32 * role + 2i32 * d) });
                let mut alive: Tile<u64, { [64] }> = ones;
                let mut path: Tile<u64, { [64] }> = zero;
                for k in 1i32..(2i32 + 6i32 * slides) {
                    let kt: Tile<i64, { [64] }> = i64s(k);
                    let tf: Tile<i64, { [64] }> = file + df * kt;
                    let tr: Tile<i64, { [64] }> = rank + dr * kt;
                    alive = alive
                        & lanes(ge_tile(tf, zero_i))
                        & lanes(lt_tile(tf, eight_i))
                        & lanes(ge_tile(tr, zero_i))
                        & lanes(lt_tile(tr, eight_i));
                    let t: Tile<i64, { [64] }> = (tr * eight_i + tf) & sixty_three;
                    let tbit: Tile<u64, { [64] }> = (one << bitcast(t)) & alive;
                    white_att = white_att | (tbit & white_l);
                    let hits_wk: Tile<u64, { [64] }> =
                        lanes(eq_tile(t, wk)) & alive & (white_l ^ ones);
                    wk_safe = wk_safe & ((path & hits_wk) | (hits_wk ^ ones));
                    path = path | tbit;
                    let blocked: Tile<u64, { [64] }> =
                        lanes(ne_tile((occ >> bitcast(t)) & one, zero));
                    alive = alive & (blocked ^ ones);
                }
            }
        }
        let wk_bit: Tile<u64, { [64] }> = one << bitcast(wk);
        let base: Tile<u64, { [64] }> =
            (occ ^ ones) & (king_reach(wk_bit) ^ ones) & lanes(eq_tile(dup, zero));
        let valid: Tile<u64, { [64] }> =
            (black & base & wk_safe) | ((black ^ ones) & base & (white_att ^ ones));
        let check: Tile<u64, { [64] }> =
            valid & ((black & white_att) | ((black ^ ones) & (wk_safe ^ ones)));
        let illegal: Tile<u64, { [64] }> = valid ^ ones;
        unsafe { store_pair(records, w, illegal, illegal) };
        unsafe { scatter(check_out, w, check) };
    }

    /// Frontier: flag every word to evaluate in the next round. These are the
    /// words that changed, or that have a child that changed, in the round that
    /// set `changed`. Parents of a word come from un-moves of the side that just
    /// moved. Sliders stop at skeleton pieces. The un-moves ignore the square of
    /// the black king, so the result is a superset.
    #[cutile::entry()]
    unsafe fn mark(dirty: *mut i32, changed: *const i32, meta: *const i32) {
        let m: i32 = unsafe { scalar_at(meta, 0i32) };
        let blk: i32 = block_id();
        let w: Tile<i64, { [64] }> = word_index(blk);
        let fresh: Tile<u64, { [64] }> = unsafe { load_lanes(changed, w) };
        if count_lanes(fresh) > 0i64 {
            let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
            let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
            let ones: Tile<u64, { [64] }> = zero - one;
            let zero_i: Tile<i64, { [64] }> = constant(0i64, shape![64]);
            let one_i: Tile<i64, { [64] }> = constant(1i64, shape![64]);
            let three_i: Tile<i64, { [64] }> = constant(3i64, shape![64]);
            let six_i: Tile<i64, { [64] }> = constant(6i64, shape![64]);
            let seven_i: Tile<i64, { [64] }> = constant(7i64, shape![64]);
            let eight_i: Tile<i64, { [64] }> = constant(8i64, shape![64]);
            let sixty_three: Tile<i64, { [64] }> = constant(63i64, shape![64]);
            let stm_bit: Tile<i64, { [64] }> = one_i << (six_i * i64s(m));
            let black_to_move: i32 = (blk >> (6i32 * m - 6i32)) & 1i32;
            unsafe { mark_flags(dirty, w, fresh) };
            if black_to_move == 0i32 {
                // The black king just moved: same skeleton, black to move.
                unsafe { mark_flags(dirty, w ^ stm_bit, fresh) };
            }
            let mut occ: Tile<u64, { [64] }> = zero;
            for i in 0i32..m {
                let sq: Tile<i64, { [64] }> = (w >> i64s(6i32 * (m - 1i32 - i))) & sixty_three;
                occ = occ | (one << bitcast(sq));
            }
            for i in 0i32..m {
                let white: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * i) };
                if white == black_to_move {
                    let role: i32 = unsafe { scalar_at(meta, 8i32 + 4i32 * i) };
                    let dirs: i32 = unsafe { scalar_at(meta, 10i32 + 4i32 * i) };
                    let slides: i32 = unsafe { scalar_at(meta, 11i32 + 4i32 * i) };
                    let shift: Tile<i64, { [64] }> = i64s(6i32 * (m - 1i32 - i));
                    let sq: Tile<i64, { [64] }> = (w >> shift) & sixty_three;
                    let file: Tile<i64, { [64] }> = sq & seven_i;
                    let rank: Tile<i64, { [64] }> = sq >> three_i;
                    for d in 0i32..dirs {
                        let df: Tile<i64, { [64] }> =
                            i64s(unsafe { scalar_at(meta, 64i32 + 16i32 * role + 2i32 * d) });
                        let dr: Tile<i64, { [64] }> =
                            i64s(unsafe { scalar_at(meta, 65i32 + 16i32 * role + 2i32 * d) });
                        let mut alive: Tile<u64, { [64] }> = fresh;
                        for k in 1i32..(2i32 + 6i32 * slides) {
                            let kt: Tile<i64, { [64] }> = i64s(k);
                            let tf: Tile<i64, { [64] }> = file + df * kt;
                            let tr: Tile<i64, { [64] }> = rank + dr * kt;
                            alive = alive
                                & lanes(ge_tile(tf, zero_i))
                                & lanes(lt_tile(tf, eight_i))
                                & lanes(ge_tile(tr, zero_i))
                                & lanes(lt_tile(tr, eight_i));
                            let t: Tile<i64, { [64] }> = (tr * eight_i + tf) & sixty_three;
                            let tbit: Tile<u64, { [64] }> = one << bitcast(t);
                            alive = alive & (lanes(ne_tile(occ & tbit, zero)) ^ ones);
                            if count_lanes(alive) > 0i64 {
                                let parent: Tile<i64, { [64] }> =
                                    (w + ((t - sq) << shift)) ^ stm_bit;
                                unsafe { mark_flags(dirty, parent, alive) };
                            }
                        }
                    }
                }
            }
        }
    }

    /// One round. MODE 0 seeds checkmates. MODE 1 derives wins and losses from
    /// `input` and commits them into `output`. In place, these are the same buffer.
    /// MODE 1 evaluates only the words that `dirty` flags. It records in
    /// `changed` which words changed. Skipped words keep both buffers valid,
    /// because they and their children did not change in the last round.
    // On the H100 NVL, occupancy 12 with 2 worker warps was the fastest.
    // Time per KQBvKR sweep: occupancy 12: 10.2 ms, occupancy 16: 10.45 ms.
    #[cutile::entry(optimization_hints = (sm_90 = (occupancy = 12, num_worker_warps_per_cta = 2,),))]
    unsafe fn sweep<const MODE: i32, const RANKED: i32>(
        output: *mut u64,
        counter: *mut i64,
        rank: *mut u16,
        input: *const u64,
        check_in: *const u64,
        lower: *const u64,
        meta: *const i32,
        round: *const u16,
        dirty: *const i32,
        changed_out: *mut i32,
        input32: *const u32,
        lower32: *const u32,
    ) {
        let blk: i32 = block_id();
        let w: Tile<i64, { [64] }> = word_index(blk);
        let active: Tile<u64, { [64] }> = unsafe { load_lanes(dirty, w) };
        if count_lanes(active) > 0i64 {
            let m: i32 = unsafe { scalar_at(meta, 0i32) };
            let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
            let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
            let ones: Tile<u64, { [64] }> = zero - one;
            let zero_i: Tile<i64, { [64] }> = constant(0i64, shape![64]);
            let one_i: Tile<i64, { [64] }> = constant(1i64, shape![64]);
            let six_i: Tile<i64, { [64] }> = constant(6i64, shape![64]);
            let sixty_three: Tile<i64, { [64] }> = constant(63i64, shape![64]);
            let m_t: Tile<i64, { [64] }> = i64s(m);
            let stm_shift: Tile<i64, { [64] }> = six_i * m_t;
            let stm_bit: Tile<i64, { [64] }> = one_i << stm_shift;
            let lower_stm_bit: Tile<i64, { [64] }> = one_i << (stm_shift - six_i);
            // Records per lower table: 2 * 64^(m - 1).
            let lower_words: Tile<i64, { [64] }> = one_i << (stm_shift - six_i + one_i);
            let black_to_move: i32 = (blk >> (6i32 * m - 6i32)) & 1i32;
            let mover_white: i32 = 1i32 - black_to_move;
            let lower_stm_h: Tile<i64, { [128] }> = i64h(1i32) << i64h(6i32 * m - 6i32);
            let lower_words_h: Tile<i64, { [128] }> = i64h(2i32) * lower_stm_h;
            let op: Tile<u64, { [64, 2] }> = unsafe { gather_pair(input, w, ones) };
            let old_win: Tile<u64, { [64] }> = pair_part(op, 0i64);
            let old_loss: Tile<u64, { [64] }> = pair_part(op, 1i64);
            let valid: Tile<u64, { [64] }> = (old_win & old_loss) ^ ones;
            let live_words: Tile<u64, { [64] }> = if MODE == 0 {
                active & lanes(ne_tile(valid, zero))
            } else {
                active & lanes(ne_tile((old_win | old_loss) ^ ones, zero))
            };
            let mut win: Tile<u64, { [64] }> = zero;
            let mut nonwin: Tile<u64, { [64] }> = zero;
            let mut any: Tile<u64, { [64] }> = zero;
            if count_lanes(live_words) > 0i64 {
                // Skeleton moves never shift bits across the black king
                // bitboard. Thus they run on its 32-bit halves: element 2 * lane
                // + half of a [128] u32 tile.
                let col: Tile<i32, { [128] }> = col_index();
                let col64: Tile<i64, { [128] }> = exti(col);
                let row: Tile<i32, { [128] }> = row_index();
                let zero_h: Tile<u32, { [128] }> = constant(0u32, shape![128]);
                let one_h: Tile<u32, { [128] }> = constant(1u32, shape![128]);
                let ones_h: Tile<u32, { [128] }> = zero_h - one_h;
                let zero_i32h: Tile<i32, { [128] }> = constant(0i32, shape![128]);
                let w_h: Tile<i64, { [128] }> = i64h(blk) * i64h(64i32) + exti(row);
                let live_h: Tile<u32, { [128] }> = halves(live_words);
                // Per-element u32 offsets of the win and loss halves of the record
                // of a lane in its block. Also the block bit of the side to move.
                let four_h: Tile<i32, { [128] }> = i32h(4i32);
                let win_off: Tile<i32, { [128] }> = four_h * row + col;
                let loss_off: Tile<i32, { [128] }> = win_off + i32h(2i32);
                let stm_blk: i32 = 1i32 << (6i32 * m - 6i32);
                let mut win_h: Tile<u32, { [128] }> = zero_h;
                let mut nonwin_h: Tile<u32, { [128] }> = zero_h;
                let mut any_h: Tile<u32, { [128] }> = zero_h;
                // Pieces 0..m-1 stand on the same squares in every lane. Lanes
                // differ only in the last piece. Trace the rays of these pieces with scalars.
                let last_white: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * (m - 1i32)) };
                let mut occ_lo: u32 = 0u32;
                let mut occ_hi: u32 = 0u32;
                let mut own_lo: u32 = 0u32;
                let mut own_hi: u32 = 0u32;
                for j in 0i32..(m - 1i32) {
                    let sq_j: i32 = (blk >> (6i32 * (m - 2i32 - j))) & 63i32;
                    occ_lo = occ_lo | lo_bit(sq_j);
                    occ_hi = occ_hi | hi_bit(sq_j);
                    let white_j: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * j) };
                    if white_j == mover_white {
                        own_lo = own_lo | lo_bit(sq_j);
                        own_hi = own_hi | hi_bit(sq_j);
                    }
                }
                // No move captures the white king (piece 0), and it always blocks.
                let wk_sq: i32 = if m > 1i32 {
                    (blk >> (6i32 * (m - 2i32))) & 63i32
                } else {
                    64i32
                };
                let block_lo: u32 = own_lo | lo_bit(wk_sq);
                let block_hi: u32 = own_hi | hi_bit(wk_sq);
                let enemy_lo: u32 = occ_lo ^ (occ_lo & block_lo);
                let enemy_hi: u32 = occ_hi ^ (occ_hi & block_hi);
                for i in 0i32..(m - 1i32) {
                    let white: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * i) };
                    if white == mover_white {
                        let role: i32 = unsafe { scalar_at(meta, 8i32 + 4i32 * i) };
                        let dirs: i32 = unsafe { scalar_at(meta, 10i32 + 4i32 * i) };
                        let slides: i32 = unsafe { scalar_at(meta, 11i32 + 4i32 * i) };
                        let sq_s: i32 = (blk >> (6i32 * (m - 2i32 - i))) & 63i32;
                        let shift_h: Tile<i64, { [128] }> = i64h(6i32 * (m - 1i32 - i));
                        // Lanes whose last piece (an enemy) this piece captures.
                        let mut cap: Tile<u32, { [128] }> = zero_h;
                        let mut cap_path: Tile<u32, { [128] }> = zero_h;
                        for d in 0i32..dirs {
                            let df: i32 =
                                unsafe { scalar_at(meta, 64i32 + 16i32 * role + 2i32 * d) };
                            let dr: i32 =
                                unsafe { scalar_at(meta, 65i32 + 16i32 * role + 2i32 * d) };
                            let mut f: i32 = sq_s & 7i32;
                            let mut r: i32 = sq_s >> 3i32;
                            let mut alive: i32 = 1i32;
                            let mut path_lo: u32 = 0u32;
                            let mut path_hi: u32 = 0u32;
                            for _k in 1i32..(2i32 + 6i32 * slides) {
                                f = f + df;
                                r = r + dr;
                                if f < 0i32 {
                                    alive = 0i32;
                                }
                                if f > 7i32 {
                                    alive = 0i32;
                                }
                                if r < 0i32 {
                                    alive = 0i32;
                                }
                                if r > 7i32 {
                                    alive = 0i32;
                                }
                                if alive != 0i32 {
                                    let t_s: i32 = r * 8i32 + f;
                                    let t_lo: u32 = lo_bit(t_s);
                                    let t_hi: u32 = hi_bit(t_s);
                                    if ((block_lo & t_lo) | (block_hi & t_hi)) != 0u32 {
                                        alive = 0i32;
                                    } else {
                                        let reach_lo: u32 = path_lo | t_lo;
                                        let reach_hi: u32 = path_hi | t_hi;
                                        let reach: Tile<u32, { [128] }> =
                                            pick(col, reach_lo, reach_hi);
                                        // The last piece blocks lanes where it stands on
                                        // the path. Where it stands on t (an enemy), this
                                        // piece captures it.
                                        let free: Tile<u32, { [128] }> =
                                            live_h & (test_bit(reach_lo, reach_hi, row) ^ ones_h);
                                        let at_t: Tile<u32, { [128] }> =
                                            live_h & lanes_h(eq_tile(row, i32h(t_s)));
                                        cap = cap | at_t;
                                        cap_path = cap_path | (at_t & reach);
                                        let captures: i32 =
                                            if ((enemy_lo & t_lo) | (enemy_hi & t_hi)) != 0u32 {
                                                1i32
                                            } else {
                                                0i32
                                            };
                                        // Captures uniform piece j: lower table j - 1.
                                        let mut jj: i32 = 1i32;
                                        if captures != 0i32 {
                                            for j in 1i32..(m - 1i32) {
                                                let sq_j: i32 =
                                                    (blk >> (6i32 * (m - 2i32 - j))) & 63i32;
                                                if sq_j == t_s {
                                                    jj = j;
                                                }
                                            }
                                        }
                                        let moved_blk: i32 =
                                            blk + ((t_s - sq_s) << (6i32 * (m - 2i32 - i)));
                                        let source: i32 = if captures != 0i32 {
                                            lower_block(moved_blk, jj, m)
                                        } else {
                                            moved_blk ^ stm_blk
                                        };
                                        let cw: Tile<u32, { [128] }> = if captures != 0i32 {
                                            unsafe { gather_block(lower32, source, win_off, free) }
                                        } else {
                                            unsafe { gather_block(input32, source, win_off, free) }
                                        };
                                        let cl: Tile<u32, { [128] }> = if captures != 0i32 {
                                            unsafe { gather_block(lower32, source, loss_off, free) }
                                        } else {
                                            unsafe { gather_block(input32, source, loss_off, free) }
                                        };
                                        if captures != 0i32 {
                                            alive = 0i32;
                                        } else {
                                            path_lo = reach_lo;
                                            path_hi = reach_hi;
                                        }
                                        let mask: Tile<u32, { [128] }> = (reach ^ ones_h) & free;
                                        win_h = win_h | (cl & (cw ^ ones_h) & mask);
                                        nonwin_h = nonwin_h | ((cw ^ ones_h) & mask);
                                        any_h = any_h | (((cw & cl) ^ ones_h) & mask);
                                    }
                                }
                            }
                        }
                        if last_white != mover_white {
                            if count_h(cap) > 0i64 {
                                // The last piece is captured: lower table m - 2.
                                let moved: Tile<i64, { [128] }> =
                                    w_h + (exti(row - i32h(sq_s)) << shift_h);
                                let source: Tile<i64, { [128] }> = lower_child(
                                    moved,
                                    i64h(m - 1i32),
                                    i64h(m),
                                    lower_words_h,
                                    lower_stm_h,
                                );
                                let cw: Tile<u32, { [128] }> =
                                    unsafe { gather_half(lower32, source, col64, cap, 0i64) };
                                let cl: Tile<u32, { [128] }> =
                                    unsafe { gather_half(lower32, source, col64, cap, 2i64) };
                                let mask: Tile<u32, { [128] }> = (cap_path ^ ones_h) & cap;
                                win_h = win_h | (cl & (cw ^ ones_h) & mask);
                                nonwin_h = nonwin_h | ((cw ^ ones_h) & mask);
                                any_h = any_h | (((cw & cl) ^ ones_h) & mask);
                            }
                        }
                    }
                }
                // The last piece stands on the square of the lane: per-lane rays.
                let i: i32 = m - 1i32;
                let white_last: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * i) };
                if white_last == mover_white {
                    let role: i32 = unsafe { scalar_at(meta, 8i32 + 4i32 * i) };
                    let dirs: i32 = unsafe { scalar_at(meta, 10i32 + 4i32 * i) };
                    let slides: i32 = unsafe { scalar_at(meta, 11i32 + 4i32 * i) };
                    let seven_h: Tile<i32, { [128] }> = i32h(7i32);
                    let file: Tile<i32, { [128] }> = row & seven_h;
                    let rank_sq: Tile<i32, { [128] }> = row >> i32h(3i32);
                    let max_file: Tile<i32, { [128] }> = seven_h - file;
                    let max_rank: Tile<i32, { [128] }> = seven_h - rank_sq;
                    let one_i32h: Tile<i32, { [128] }> = i32h(1i32);
                    for d in 0i32..dirs {
                        let df: i32 = unsafe { scalar_at(meta, 64i32 + 16i32 * role + 2i32 * d) };
                        let dr: i32 = unsafe { scalar_at(meta, 65i32 + 16i32 * role + 2i32 * d) };
                        // Steps to the board edge (leaper steps are at most 2 per axis).
                        let mut room: Tile<i32, { [128] }> = seven_h;
                        if df == 1i32 {
                            room = min_tile(room, max_file);
                        }
                        if df == 2i32 {
                            room = min_tile(room, max_file >> one_i32h);
                        }
                        if df == -1i32 {
                            room = min_tile(room, file);
                        }
                        if df == -2i32 {
                            room = min_tile(room, file >> one_i32h);
                        }
                        if dr == 1i32 {
                            room = min_tile(room, max_rank);
                        }
                        if dr == 2i32 {
                            room = min_tile(room, max_rank >> one_i32h);
                        }
                        if dr == -1i32 {
                            room = min_tile(room, rank_sq);
                        }
                        if dr == -2i32 {
                            room = min_tile(room, rank_sq >> one_i32h);
                        }
                        let mut alive: Tile<u32, { [128] }> = live_h;
                        let mut path: Tile<u32, { [128] }> = zero_h;
                        // A ray captures at most once, at its end. Collect the
                        // capture square and path. Look up all captures after the loop.
                        let mut cap: Tile<u32, { [128] }> = zero_h;
                        let mut cap_t: Tile<i32, { [128] }> = zero_i32h;
                        let mut cap_path: Tile<u32, { [128] }> = zero_h;
                        for k in 1i32..(2i32 + 6i32 * slides) {
                            alive = alive & lanes_h(le_tile(i32h(k), room));
                            let t: Tile<i32, { [128] }> = row + i32h(k * (df + 8i32 * dr));
                            let tbit: Tile<u32, { [128] }> = half_bit(t, col);
                            let hit_own: Tile<u32, { [128] }> = test_bit(block_lo, block_hi, t);
                            let hit_enemy: Tile<u32, { [128] }> = test_bit(enemy_lo, enemy_hi, t);
                            let moving: Tile<u32, { [128] }> = alive & (hit_own ^ ones_h);
                            let quiet: Tile<u32, { [128] }> = moving & (hit_enemy ^ ones_h);
                            let capture: Tile<u32, { [128] }> = moving & hit_enemy;
                            cap = cap | capture;
                            cap_t = select(ne_tile(capture, zero_h), t, cap_t);
                            cap_path = cap_path | (capture & (path | tbit));
                            let t_off: Tile<i32, { [128] }> = four_h * t + col;
                            let cw: Tile<u32, { [128] }> =
                                unsafe { gather_block(input32, blk ^ stm_blk, t_off, quiet) };
                            let cl: Tile<u32, { [128] }> = unsafe {
                                gather_block(input32, blk ^ stm_blk, t_off + i32h(2i32), quiet)
                            };
                            let mask: Tile<u32, { [128] }> = ((path | tbit) ^ ones_h) & quiet;
                            win_h = win_h | (cl & (cw ^ ones_h) & mask);
                            nonwin_h = nonwin_h | ((cw ^ ones_h) & mask);
                            any_h = any_h | (((cw & cl) ^ ones_h) & mask);
                            path = path | (tbit & alive);
                            alive = alive & (test_bit(occ_lo, occ_hi, t) ^ ones_h);
                        }
                        if count_h(cap) > 0i64 {
                            // Captured skeleton piece j (a uniform piece): lower
                            // table j - 1, without piece j.
                            let mut j_t: Tile<i64, { [128] }> = i64h(0i32);
                            for j in 1i32..(m - 1i32) {
                                let sq_j: i32 = (blk >> (6i32 * (m - 2i32 - j))) & 63i32;
                                j_t = select(eq_tile(cap_t, i32h(sq_j)), i64h(j), j_t);
                            }
                            let moved: Tile<i64, { [128] }> = w_h + exti(cap_t - row);
                            let source: Tile<i64, { [128] }> =
                                lower_child(moved, j_t, i64h(m), lower_words_h, lower_stm_h);
                            let cw: Tile<u32, { [128] }> =
                                unsafe { gather_half(lower32, source, col64, cap, 0i64) };
                            let cl: Tile<u32, { [128] }> =
                                unsafe { gather_half(lower32, source, col64, cap, 2i64) };
                            let mask: Tile<u32, { [128] }> = (cap_path ^ ones_h) & cap;
                            win_h = win_h | (cl & (cw ^ ones_h) & mask);
                            nonwin_h = nonwin_h | ((cw ^ ones_h) & mask);
                            any_h = any_h | (((cw & cl) ^ ones_h) & mask);
                        }
                    }
                }
                win = win | join(win_h);
                nonwin = nonwin | join(nonwin_h);
                any = any | join(any_h);
                // Black king moves inside the word, and black king captures.
                if black_to_move == 1i32 {
                    let cp: Tile<u64, { [64, 2] }> =
                        unsafe { gather_pair(input, w ^ stm_bit, live_words) };
                    let cw: Tile<u64, { [64] }> = pair_part(cp, 0i64);
                    let cl: Tile<u64, { [64] }> = pair_part(cp, 1i64);
                    win = win | king_reach(cl & (cw ^ ones));
                    nonwin = nonwin | king_reach(cw ^ ones);
                    any = any | king_reach((cw & cl) ^ ones);
                    for j in 1i32..m {
                        let white: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * j) };
                        if white == 1i32 {
                            let low_shift: Tile<i64, { [64] }> = i64s(6i32 * (m - 1i32 - j));
                            let t: Tile<i64, { [64] }> = (w >> low_shift) & sixty_three;
                            let low_mask: Tile<i64, { [64] }> = (one_i << low_shift) - one_i;
                            let hi: Tile<i64, { [64] }> = w >> (low_shift + six_i);
                            let low_word: Tile<i64, { [64] }> =
                                ((hi << low_shift) | (w & low_mask)) ^ lower_stm_bit;
                            let lower_index: Tile<i64, { [64] }> =
                                i64s(j - 1i32) * lower_words + low_word;
                            let lp: Tile<u64, { [64, 2] }> =
                                unsafe { gather_pair(lower, lower_index, live_words) };
                            let lw: Tile<u64, { [64] }> = pair_part(lp, 0i64);
                            let ll: Tile<u64, { [64] }> = pair_part(lp, 1i64);
                            let tu: Tile<u64, { [64] }> = bitcast(t);
                            let sources: Tile<u64, { [64] }> = king_reach(one << tu);
                            let bit_l: Tile<u64, { [64] }> =
                                lanes(ne_tile(((ll & (lw ^ ones)) >> tu) & one, zero));
                            let bit_n: Tile<u64, { [64] }> =
                                lanes(ne_tile(((lw ^ ones) >> tu) & one, zero));
                            let bit_v: Tile<u64, { [64] }> =
                                lanes(ne_tile((((lw & ll) ^ ones) >> tu) & one, zero));
                            win = win | (sources & bit_l);
                            nonwin = nonwin | (sources & bit_n);
                            any = any | (sources & bit_v);
                        }
                    }
                }
            }
            let live: Tile<u64, { [64] }> = valid & active;
            let mut new_win: Tile<u64, { [64] }> = old_win | (win & live);
            let mut new_loss: Tile<u64, { [64] }> = old_loss | (live & any & (nonwin ^ ones));
            if MODE == 0 {
                let check: Tile<u64, { [64] }> = unsafe { gather(check_in, w) };
                new_win = old_win;
                new_loss = old_loss | (valid & (any ^ ones) & check);
            }
            unsafe { store_pair(output, w, new_win, new_loss) };
            // Sets are monotone and illegal bits never change, so XOR is exactly
            // the positions decided in this round.
            let fresh: Tile<u64, { [64] }> = (new_win ^ old_win) | (new_loss ^ old_loss);
            let changed: Tile<i64, { [64] }> = select(ne_tile(fresh, zero), one_i, zero_i);
            let total: Tile<i64, { [] }> = reduce_sum(changed, 0);
            let block_changed: i64 = tile_to_scalar(total);
            if block_changed > 0i64 {
                unsafe { store_lanes(changed_out, w, fresh) };
                if RANKED == 1 {
                    // Proof rank = this round, for every position decided now.
                    let value: u16 = unsafe { scalar_u16_at(round, 0i32) };
                    let fresh2: Tile<u64, { [64, 64] }> =
                        fresh.reshape(shape![64, 1]).broadcast(shape![64, 64]);
                    let column: Tile<u64, { [64] }> = iota(shape![64]);
                    let column: Tile<u64, { [64, 64] }> =
                        column.reshape(shape![1, 64]).broadcast(shape![64, 64]);
                    let one2: Tile<u64, { [64, 64] }> = constant(1u64, shape![64, 64]);
                    let zero2: Tile<u64, { [64, 64] }> = constant(0u64, shape![64, 64]);
                    let set: Tile<bool, { [64, 64] }> = ne_tile((fresh2 >> column) & one2, zero2);
                    let row: Tile<i64, { [64, 64] }> =
                        w.reshape(shape![64, 1]).broadcast(shape![64, 64]);
                    let col: Tile<i64, { [64] }> = iota(shape![64]);
                    let col: Tile<i64, { [64, 64] }> =
                        col.reshape(shape![1, 64]).broadcast(shape![64, 64]);
                    let width: Tile<i64, { [64, 64] }> = constant(64i64, shape![64, 64]);
                    let base: PointerTile<*mut u16, { [] }> = pointer_to_tile(rank);
                    let base: PointerTile<*mut u16, { [1, 1] }> = base.reshape(shape![1, 1]);
                    let base: PointerTile<*mut u16, { [64, 64] }> = base.broadcast(shape![64, 64]);
                    let ptrs: PointerTile<*mut u16, { [64, 64] }> =
                        base.offset_tile(row * width + col);
                    let values: Tile<u16, { [64, 64] }> = broadcast_scalar(value, shape![64, 64]);
                    // SAFETY: indices are below words * 64, the length of `rank`.
                    unsafe {
                        store_ptr_tko(
                            ptrs,
                            values,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            Some(set),
                            None,
                            Latency::<0>,
                        )
                    };
                }
                let total: Tile<i64, { [1] }> = total.reshape(shape![1]);
                let cptr: PointerTile<*mut i64, { [] }> = pointer_to_tile(counter);
                let cptr: PointerTile<*mut i64, { [1] }> = cptr.reshape(shape![1]);
                let _old: (Tile<i64, { [1] }>, Token) = unsafe {
                    atomic_rmw_tko(
                        cptr,
                        total,
                        atomic::Add,
                        ordering::Relaxed,
                        scope::Device,
                        None,
                        None,
                    )
                };
            }
        }
    }
}

fn role_code(role: pawnless::Role) -> i32 {
    match role {
        pawnless::Role::King => 0,
        pawnless::Role::Queen => 1,
        pawnless::Role::Rook => 2,
        pawnless::Role::Bishop => 3,
        pawnless::Role::Knight => 4,
    }
}

fn meta_table(material: &Material) -> Vec<i32> {
    let mut meta = vec![0i32; META_LEN];
    meta[0] = material.skeleton_len() as i32;
    for (i, piece) in material.skeleton.iter().enumerate() {
        let (directions, slides) = piece.role.movement();
        meta[8 + 4 * i..12 + 4 * i].copy_from_slice(&[
            role_code(piece.role),
            i32::from(piece.white),
            directions.len() as i32,
            i32::from(slides),
        ]);
    }
    for role in [
        pawnless::Role::King,
        pawnless::Role::Queen,
        pawnless::Role::Rook,
        pawnless::Role::Bishop,
        pawnless::Role::Knight,
    ] {
        let (directions, _) = role.movement();
        for (d, &(df, dr)) in directions.iter().enumerate() {
            let at = 64 + 2 * (8 * role_code(role) as usize + d);
            meta[at..at + 2].copy_from_slice(&[i32::from(df), i32::from(dr)]);
        }
    }
    meta
}

/// Device records `[win | illegal, loss | illegal]` of a solved table.
fn records(table: &Table) -> Vec<u64> {
    let mut out = Vec::with_capacity(2 * table.valid.len());
    for w in 0..table.valid.len() {
        let illegal = !table.valid[w];
        out.push(table.win[w] | illegal);
        out.push(table.loss[w] | illegal);
    }
    out
}

type Error = Box<dyn std::error::Error>;

/// Largest grid X in use. Ids above 2^24 on one axis are unreliable in the pinned
/// cuTile. Thus larger launches add a Y dimension. Word counts are powers of two.
const GRID_X: usize = 1 << 20;

fn launch_grid(words: usize) -> (u32, u32, u32) {
    let blocks = words / BLOCK;
    if blocks <= GRID_X {
        (blocks as u32, 1, 1)
    } else {
        (GRID_X as u32, (blocks / GRID_X) as u32, 1)
    }
}

/// Stream-ordered device allocation. cuTile tensors must have fewer than 2^31
/// elements, but six-piece state holds more. Thus solver buffers use raw driver
/// allocations, and the kernels take raw pointers.
struct DeviceVec<T> {
    ptr: cutile::cuda_core::sys::CUdeviceptr,
    len: usize,
    stream: Arc<cutile::cuda_core::Stream>,
    _type: std::marker::PhantomData<T>,
}

impl<T: Copy + Default> DeviceVec<T> {
    fn zeroed(stream: &Arc<cutile::cuda_core::Stream>, len: usize) -> Result<Self, Error> {
        Self::filled(stream, 0, len)
    }

    fn filled(
        stream: &Arc<cutile::cuda_core::Stream>,
        byte: u8,
        len: usize,
    ) -> Result<Self, Error> {
        let bytes = len.max(1) * std::mem::size_of::<T>();
        // SAFETY: valid stream. The allocation is initialized before any use.
        let ptr = unsafe { cutile::cuda_core::malloc_async(bytes, stream)? };
        unsafe { cutile::cuda_core::memset_d8_async(ptr, byte, bytes, stream)? };
        Ok(DeviceVec {
            ptr,
            len,
            stream: stream.clone(),
            _type: std::marker::PhantomData,
        })
    }

    fn from_host(stream: &Arc<cutile::cuda_core::Stream>, data: &[T]) -> Result<Self, Error> {
        let v = Self::zeroed(stream, data.len())?;
        // SAFETY: data and the allocation both hold data.len() elements. The
        // stream is synchronized before the caller can drop `data`.
        unsafe {
            cutile::cuda_core::memcpy_htod_async(v.ptr, data.as_ptr(), data.len(), stream)?;
            stream.synchronize()?;
        }
        Ok(v)
    }

    fn copy_from_host(&mut self, data: &[T]) -> Result<(), Error> {
        if data.len() != self.len {
            return Err("host and device lengths differ".into());
        }
        // SAFETY: both hold len elements. The stream is synchronized before `data` is released.
        unsafe {
            cutile::cuda_core::memcpy_htod_async(
                self.ptr,
                data.as_ptr(),
                data.len(),
                &self.stream,
            )?;
            self.stream.synchronize()?;
        }
        Ok(())
    }

    fn to_host(&self) -> Result<Vec<T>, Error> {
        let mut out = vec![T::default(); self.len];
        self.read_into(0, &mut out)?;
        Ok(out)
    }

    /// Copy elements `offset..offset + out.len()` to the host.
    fn read_into(&self, offset: usize, out: &mut [T]) -> Result<(), Error> {
        if offset + out.len() > self.len {
            return Err("device read out of range".into());
        }
        let at = self.ptr + (offset * std::mem::size_of::<T>()) as u64;
        // SAFETY: the range lies inside the allocation (checked above) and the
        // stream is synchronized before `out` is released.
        unsafe {
            cutile::cuda_core::memcpy_dtoh_async(out.as_mut_ptr(), at, out.len(), &self.stream)?;
            self.stream.synchronize()?;
        }
        Ok(())
    }

    fn ptr(&self) -> cutile::tile_kernel::DevicePointer<T> {
        // SAFETY: `ptr` is a live allocation of `len` values of T.
        unsafe { cutile::tile_kernel::DevicePointer::from_cu_deviceptr(self.ptr) }
    }

    /// The same allocation as elements of type U.
    fn view<U>(&self) -> cutile::tile_kernel::DevicePointer<U> {
        // SAFETY: `ptr` is a live allocation. Kernels index it in range for U.
        unsafe { cutile::tile_kernel::DevicePointer::from_cu_deviceptr(self.ptr) }
    }

    fn bytes(&self) -> usize {
        self.len.max(1) * std::mem::size_of::<T>()
    }
}

impl<T> Drop for DeviceVec<T> {
    fn drop(&mut self) {
        // SAFETY: malloc_async allocated it on this stream. Stream order puts the
        // free after every kernel that used it.
        let _ = unsafe { cutile::cuda_core::free_async(self.ptr, &self.stream) };
    }
}

struct Gpu {
    stream: Arc<cutile::cuda_core::Stream>,
    /// After a round that changed at most this fraction of the words, rounds
    /// sweep only the frontier. The value comes from `TB_FRONTIER`, and 0 disables it.
    frontier: f64,
}

impl Gpu {
    /// Solve one material on the GPU. Returns the table and phase records.
    ///
    /// `ranked`: synchronous double-buffered rounds with device proof ranks.
    /// Otherwise, the solver updates WDL in place. This is sound because the
    /// sets are monotone. It halves the state but gives no proof ranks.
    ///
    /// With `stored`, the solver downloads the stored words (white king in the
    /// triangle) and their proof ranks as a `Canonical`. Then the full table, if
    /// `full`, has no rank plane. The device checks symmetry and counts.
    fn solve(
        &self,
        material: &Material,
        lower: &[Option<&Table>],
        ranked: bool,
        stored: Option<&Layout>,
        full: bool,
    ) -> Result<Solution, Error> {
        let start = Instant::now();
        let words = material.words();
        // Lower tables in skeleton order k = 1..m, each 2 * 64^(m - 1) records.
        let mut lower_data: Vec<u64> = Vec::new();
        for (k, table) in lower.iter().enumerate().skip(1) {
            let t = table.ok_or(format!("missing lower table for skeleton piece {k}"))?;
            lower_data.extend(records(t));
        }
        let meta = DeviceVec::from_host(&self.stream, &meta_table(material))?;
        let lower_dev = DeviceVec::from_host(&self.stream, &lower_data)?;
        drop(lower_data);
        let state = [
            DeviceVec::<u64>::zeroed(&self.stream, 2 * words)?,
            DeviceVec::<u64>::zeroed(&self.stream, if ranked { 2 * words } else { 1 })?,
        ];
        let check = DeviceVec::<u64>::zeroed(&self.stream, words)?;
        // UNRESOLVED is 0xffff: fill every byte with 0xff. Unranked solves pass
        // a one-element placeholder that the kernel never writes.
        let rank =
            DeviceVec::<u16>::filled(&self.stream, 0xff, if ranked { words * 64 } else { 1 })?;
        let mut round_value = DeviceVec::<u16>::zeroed(&self.stream, 1)?;
        // Per-word frontier flags: `dirty` gates the next sweep, and every
        // sweep writes `changed`.
        let dirty = DeviceVec::<i32>::filled(&self.stream, 1, words)?;
        let changed_flags = DeviceVec::<i32>::zeroed(&self.stream, words)?;
        let device_bytes = state[0].bytes()
            + state[1].bytes()
            + check.bytes()
            + lower_dev.bytes()
            + rank.bytes()
            + dirty.bytes()
            + changed_flags.bytes();
        let grid = launch_grid(words);
        let device = self.stream.device();
        let (event_start, event_mid, event_end) = (
            device.new_event()?,
            device.new_event()?,
            device.new_event()?,
        );
        let prepare_seconds = start.elapsed().as_secs_f64();
        let t0 = Instant::now();
        // SAFETY: every pointer addresses a live device buffer sized for this
        // material. Kernels touch words < `words` and lower records in range.
        unsafe {
            kernel::statics(state[0].ptr(), check.ptr(), meta.ptr())
                .grid(grid)
                .sync_on(&self.stream)?;
        }
        if ranked {
            // Both buffers start from the statics records.
            // SAFETY: equal-length device buffers, stream ordered.
            unsafe {
                cutile::cuda_core::memcpy_dtod_async::<u64>(
                    state[1].ptr,
                    state[0].ptr,
                    2 * words,
                    &self.stream,
                )?;
            }
        }
        let statics_seconds = t0.elapsed().as_secs_f64();
        let frontier_limit = (self.frontier * words as f64) as i64;
        let mut last_changed = i64::MAX;
        let mut rounds = 0u32;
        let mut records_log = Vec::new();
        for round in 0u32.. {
            // Round r writes proof rank r. UNRESOLVED is not a rank.
            if ranked && round >= u32::from(UNRESOLVED) {
                return Err("proof rank exceeds the 16-bit representation".into());
            }
            let r0 = Instant::now();
            // Ranked: round r reads state[(r + 1) % 2] and writes state[r % 2].
            let (src, dst) = if ranked {
                ((round as usize + 1) % 2, round as usize % 2)
            } else {
                (0, 0)
            };
            let counter = DeviceVec::<i64>::zeroed(&self.stream, 1)?;
            round_value.copy_from_host(&[round as u16])?;
            let mode = if round == 0 { "0" } else { "1" };
            // Round 1 reads the lower tables for the first time: always full.
            let frontier = round >= 2 && last_changed <= frontier_limit;
            event_start.record(&self.stream)?;
            if frontier {
                // SAFETY: flag buffers hold one i32 per word. Stream ordered.
                unsafe {
                    cutile::cuda_core::memset_d8_async(dirty.ptr, 0, dirty.bytes(), &self.stream)?;
                    kernel::mark(dirty.ptr(), changed_flags.ptr(), meta.ptr())
                        .grid(grid)
                        .sync_on(&self.stream)?;
                }
            } else if round >= 2 {
                // SAFETY: as above. Nonzero bytes flag every word.
                unsafe {
                    cutile::cuda_core::memset_d8_async(dirty.ptr, 1, dirty.bytes(), &self.stream)?;
                }
            }
            if round >= 1 {
                unsafe {
                    cutile::cuda_core::memset_d8_async(
                        changed_flags.ptr,
                        0,
                        changed_flags.bytes(),
                        &self.stream,
                    )?;
                }
            }
            event_mid.record(&self.stream)?;
            // SAFETY: as above. In place, input and output are the same buffer.
            // Monotone updates make concurrent reads of old or new values sound.
            unsafe {
                kernel::sweep(
                    state[dst].ptr(),
                    counter.ptr(),
                    rank.ptr(),
                    state[src].ptr(),
                    check.ptr(),
                    lower_dev.ptr(),
                    meta.ptr(),
                    round_value.ptr(),
                    dirty.ptr(),
                    changed_flags.ptr(),
                    state[src].view::<u32>(),
                    lower_dev.view::<u32>(),
                )
                .generics(vec![mode.into(), if ranked { "1" } else { "0" }.into()])
                .grid(grid)
                .sync_on(&self.stream)?;
            }
            event_end.record(&self.stream)?;
            event_end.synchronize()?;
            let mark_event_ms = event_start.elapsed_time(&event_mid)?;
            let sweep_event_ms = event_mid.elapsed_time(&event_end)?;
            let changed = counter.to_host()?[0];
            records_log.push(json!({
                "round": round, "changed_words": changed, "frontier": frontier,
                "host_seconds": r0.elapsed().as_secs_f64(),
                "mark_event_ms": mark_event_ms, "sweep_event_ms": sweep_event_ms,
            }));
            if round > 0 && changed == 0 {
                break;
            }
            last_changed = changed;
            rounds = round;
        }
        let rounds_seconds = t0.elapsed().as_secs_f64() - statics_seconds;
        let last = if ranked { rounds as usize % 2 } else { 0 };
        let t1 = Instant::now();
        let finish = DeviceVec::<i64>::zeroed(&self.stream, 7)?;
        // SAFETY: `state[last]` holds 2 * words records, and `finish` holds 7 counters.
        unsafe {
            kernel::finish(finish.ptr(), state[last].ptr(), meta.ptr())
                .grid(grid)
                .sync_on(&self.stream)?;
        }
        let f = finish.to_host()?;
        let counts: [Counts; 2] = std::array::from_fn(|turn| {
            let [legal, win, loss] = [0, 1, 2].map(|k| f[1 + 3 * turn + k] as u64);
            Counts {
                legal,
                win,
                loss,
                draw: legal - win - loss,
            }
        });
        let finish_seconds = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let canonical = match stored {
            None => None,
            Some(layout) => {
                // Stored words form 20 contiguous runs of the full layout.
                let per = layout.words() / 20;
                let mut flat = vec![0u64; 2 * layout.words()];
                for (run, dst) in flat.chunks_mut(2 * per).enumerate() {
                    state[last].read_into(2 * layout.full_word(run * per), dst)?;
                }
                let mut stored_rank = Vec::new();
                if ranked {
                    stored_rank = vec![0u16; layout.words() * 64];
                    for (run, dst) in stored_rank.chunks_mut(per * 64).enumerate() {
                        rank.read_into(layout.full_word(run * per) * 64, dst)?;
                    }
                }
                Some(Canonical {
                    material: material.clone(),
                    rounds,
                    records: flat.as_chunks::<2>().0.to_vec(),
                    rank: stored_rank,
                    counts: Some(counts),
                })
            }
        };
        let table = if full {
            let wl = state[last].to_host()?;
            let mut table = Table {
                material: material.clone(),
                valid: vec![0; words],
                win: vec![0; words],
                loss: vec![0; words],
                rank: if ranked && stored.is_none() {
                    rank.to_host()?
                } else {
                    Vec::new()
                },
                rounds,
            };
            // Records to planes, in parallel chunks.
            let chunk = words.div_ceil(CONVERT_THREADS).max(1);
            std::thread::scope(|scope| {
                let parts = table
                    .valid
                    .chunks_mut(chunk)
                    .zip(table.win.chunks_mut(chunk))
                    .zip(table.loss.chunks_mut(chunk))
                    .zip(wl.chunks(2 * chunk));
                for (((valid, win), loss), rec) in parts {
                    scope.spawn(move || {
                        for (i, pair) in rec.as_chunks::<2>().0.iter().enumerate() {
                            let ok = !(pair[0] & pair[1]);
                            valid[i] = ok;
                            win[i] = pair[0] & ok;
                            loss[i] = pair[1] & ok;
                        }
                    });
                }
            });
            Some(table)
        } else {
            None
        };
        let record = json!({
            "mode": if ranked { "synchronous-ranked" } else { "in-place-wdl" },
            "words": words, "device_bytes_allocated": device_bytes,
            "prepare_seconds": prepare_seconds,
            "statics_seconds_including_jit": statics_seconds,
            "rounds_seconds_total": rounds_seconds,
            "finish_seconds": finish_seconds,
            "download_seconds": t2.elapsed().as_secs_f64(),
            "rounds": records_log,
        });
        Ok(Solution {
            table,
            canonical,
            rounds,
            counts,
            violations: f[0] as u64,
            record,
        })
    }
}

/// A finished solve: the full working-layout table and/or its stored form.
struct Solution {
    table: Option<Table>,
    canonical: Option<Canonical>,
    rounds: u32,
    counts: [Counts; 2],
    /// Words whose records differ from a symmetric image (zero when correct).
    violations: u64,
    record: serde_json::Value,
}

/// Every pawnless material of at most `pieces` pieces that a complete set
/// must have. The side with more pieces is white. For equal counts, white is
/// the side whose pieces come first in QRBN order. The list is closed under
/// captures, with lower materials first.
fn all_materials(pieces: usize) -> Result<Vec<Material>, Error> {
    const ROLES: [char; 4] = ['Q', 'R', 'B', 'N'];
    fn multisets(n: usize, from: usize, prefix: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if n == 0 {
            out.push(prefix.clone());
            return;
        }
        for r in from..ROLES.len() {
            prefix.push(r);
            multisets(n - 1, r, prefix, out);
            prefix.pop();
        }
    }
    let text = |set: &[usize]| set.iter().map(|&r| ROLES[r]).collect::<String>();
    let mut pending = Vec::new();
    for total in 0..=pieces.saturating_sub(2) {
        for white in total.div_ceil(2)..=total {
            let (mut ws, mut bs) = (Vec::new(), Vec::new());
            multisets(white, 0, &mut Vec::new(), &mut ws);
            multisets(total - white, 0, &mut Vec::new(), &mut bs);
            for w in &ws {
                for b in &bs {
                    if 2 * white > total || w <= b {
                        pending.push(format!("K{}vK{}", text(w), text(b)));
                    }
                }
            }
        }
    }
    let mut all = std::collections::BTreeMap::new();
    while let Some(name) = pending.pop() {
        if all.contains_key(&name) {
            continue;
        }
        let m = Material::parse(&name)?;
        for k in 1..m.skeleton_len() {
            pending.push(m.without(k)?.name());
        }
        all.insert(name, m);
    }
    let mut list: Vec<Material> = all.into_values().collect();
    list.sort_by_key(|m| (m.skeleton_len(), m.name()));
    Ok(list)
}

/// A solved table on its way to the compressed file.
struct Job {
    stored: Canonical,
    path: PathBuf,
    record: serde_json::Value,
}

/// Background writer of the v3 file. It writes to a temporary name and renames
/// the file when it is complete. The device already checked symmetry.
fn write_jobs(
    jobs: std::sync::mpsc::Receiver<Job>,
    level: i32,
    threads: usize,
) -> tablebase_core::Result<Vec<serde_json::Value>> {
    let mut out = Vec::new();
    for mut job in jobs {
        let t = Instant::now();
        let partial = job.path.with_extension("tbz.partial");
        let file = File::create(&partial).map_err(|e| e.to_string())?;
        let options = packed::WriteOptions {
            level,
            ..Default::default()
        };
        let header = packed::write(
            &job.stored,
            BufWriter::new(file),
            GENERATOR,
            options,
            threads,
        )?;
        drop(job.stored);
        std::fs::rename(&partial, &job.path).map_err(|e| e.to_string())?;
        let r = &mut job.record;
        r["pack_seconds"] = t.elapsed().as_secs_f64().into();
        r["bytes"] = std::fs::metadata(&job.path)
            .map_err(|e| e.to_string())?
            .len()
            .into();
        r["sections"] = json!(header.sections);
        r["payload_sha256"] = header.payload_sha256.into();
        r["content_sha256"] = header.content_sha256.into();
        eprintln!(
            "{} rounds {} sweeps {:.2}s packed {} bytes",
            header.material, header.rounds, r["timing"]["rounds_seconds_total"], r["bytes"]
        );
        out.push(job.record);
    }
    Ok(out)
}

/// Solve every material of at most `pieces` pieces in one process and write
/// compressed v3 tables. Tables that later materials capture into stay in
/// memory. The solver reuses finished v3 files from this exact clean build.
fn solve_all(
    gpu: &Gpu,
    pieces: usize,
    output: &std::path::Path,
    level: i32,
    threads: usize,
) -> Result<serde_json::Value, Error> {
    if pieces > 5 {
        // A ranked six-piece solve uses more than 300 GB of device memory.
        return Err(format!(
            "--all {pieces}: ranked solves fit in device memory up to 5 pieces. \
             Solve a six-piece tree with `gpu_solve MATERIAL OUTPUT_DIRECTORY`."
        )
        .into());
    }
    let materials = all_materials(pieces)?;
    let (send, receive) = std::sync::mpsc::sync_channel::<Job>(2);
    let writer = std::thread::spawn(move || write_jobs(receive, level, threads));
    let mut resumed = Vec::new();
    let solved = solve_all_into(gpu, &materials, output, threads, &send, &mut resumed);
    drop(send);
    // Let the writer finish the tables it has before an error stops the process.
    let tables = writer.join().map_err(|_| "writer thread panicked")?;
    solved?;
    Ok(json!({"materials": materials.len(), "resumed_verified_tables": resumed, "tables": tables?}))
}

fn solve_all_into(
    gpu: &Gpu,
    materials: &[Material],
    output: &std::path::Path,
    threads: usize,
    send: &std::sync::mpsc::SyncSender<Job>,
    resumed: &mut Vec<serde_json::Value>,
) -> Result<(), Error> {
    let max_skeleton = materials
        .iter()
        .map(Material::skeleton_len)
        .max()
        .unwrap_or(1);
    let mut solved = Solved::new();
    for m in materials {
        let name = m.name();
        let path = output.join(format!("{name}.tbz"));
        let needed_below = m.skeleton_len() < max_skeleton;
        if path.exists() && !GENERATOR.ends_with("+dirty") {
            let packed = PackedTable::open(&path, true)?;
            if packed.header.generator == GENERATOR {
                if needed_below {
                    solved.insert(name.clone(), Arc::new(packed.decode(threads)?));
                }
                resumed.push(
                    json!({"material": name, "payload_sha256": packed.header.payload_sha256}),
                );
                continue;
            }
        }
        let layout = Layout::new(m);
        let solution = {
            let lower: Vec<Option<&Table>> = (0..m.skeleton_len())
                .map(|k| match k {
                    0 => Ok(None),
                    _ => m
                        .without(k)
                        .map(|l| solved.get(&l.name()).map(|t| t.as_ref())),
                })
                .collect::<tablebase_core::Result<_>>()?;
            // Only a material that a later material captures into must keep the full table.
            gpu.solve(m, &lower, true, Some(&layout), needed_below)?
        };
        if solution.violations != 0 {
            return Err(format!("{name}: {} symmetry-violating words", solution.violations).into());
        }
        let counts = solution.counts;
        let record = json!({
            "material": name, "rounds": solution.rounds, "timing": solution.record,
            "symmetry_violating_words": solution.violations,
            "counts_white_to_move": counts[0], "counts_black_to_move": counts[1], "file": path,
        });
        if let Some(table) = solution.table {
            solved.insert(name, Arc::new(table));
        }
        let stored = solution.canonical.ok_or("stored form missing")?;
        if send
            .send(Job {
                stored,
                path,
                record,
            })
            .is_err()
        {
            break;
        }
    }
    Ok(())
}

fn main() -> Result<(), Error> {
    let wall = Instant::now();
    let mut args = std::env::args().skip(1).peekable();
    let usage = "usage: gpu_solve MATERIAL OUTPUT_DIRECTORY [MAX_RANKED_SKELETON]\n       \
                 gpu_solve --all PIECES OUTPUT_DIRECTORY [ZSTD_LEVEL [PACK_THREADS]]";
    let all = args.next_if(|a| a == "--all").is_some();
    let target = args.next().ok_or(usage)?;
    let output = PathBuf::from(args.next().ok_or(usage)?);
    std::fs::create_dir_all(&output)?;
    // Compiled kernels persist across runs (keyed by source, generics, toolkit).
    jit_cache::enable(Arc::new(FileSystemJitStore::default_location()?));
    let init = Instant::now();
    let device = cutile::cuda_core::Device::new(0)?;
    let gpu = Gpu {
        stream: device.new_stream()?,
        frontier: std::env::var("TB_FRONTIER").map_or(Ok(0.03), |v| v.parse())?,
    };
    let device_init_seconds = init.elapsed().as_secs_f64();
    let mut report = if all {
        let level = args.next().map_or(Ok(19), |v| v.parse())?;
        let threads = args.next().map_or(Ok(20), |v| v.parse())?;
        solve_all(&gpu, target.parse()?, &output, level, threads)?
    } else {
        solve_one_tree(&gpu, &Material::parse(&target)?, &output, args.next())?
    };
    report["solver"] = "gpu-cutile".into();
    report["generator"] = GENERATOR.into();
    report["device"] = std::env::var("CUDA_VISIBLE_DEVICES")
        .unwrap_or_default()
        .into();
    report["device_init_seconds"] = device_init_seconds.into();
    report["wall_seconds"] = wall.elapsed().as_secs_f64().into();
    println!("{report}");
    Ok(())
}

/// Solve `material` and its capture dependencies, and write v2 files.
fn solve_one_tree(
    gpu: &Gpu,
    material: &Material,
    output: &std::path::Path,
    max_ranked: Option<String>,
) -> Result<serde_json::Value, Error> {
    // Proof ranks use 128 bytes per word: keep them up to five pieces.
    let max_ranked: usize = max_ranked.map_or(Ok(4), |v| v.parse())?;
    let mut records_out = Vec::new();
    let mut solve_one = |m: &Material, lower: &[Option<&Table>]| -> tablebase_core::Result<Table> {
        let ranked = m.skeleton_len() <= max_ranked;
        let solution = gpu
            .solve(m, lower, ranked, None, true)
            .map_err(|e| e.to_string())?;
        let table = solution.table.ok_or("full table missing")?;
        let timing = solution.record;
        let counts = table.counts();
        if counts != solution.counts || solution.violations != 0 {
            return Err(format!(
                "{}: device checks disagree (counts {:?} vs {counts:?}, {} violations)",
                m.name(),
                solution.counts,
                solution.violations
            ));
        }
        let t = Instant::now();
        let path = output.join(format!("{}.gpu.tbp", m.name()));
        // A resumed run reads each .gpu.tbp. Thus write a full file or none.
        let partial = path.with_extension("tbp.partial");
        let file = File::create(&partial).map_err(|e| e.to_string())?;
        let header = table
            .write(BufWriter::new(file), GENERATOR)
            .and_then(|h| {
                std::fs::rename(&partial, &path)
                    .map(|()| h)
                    .map_err(|e| e.to_string())
            })
            .inspect_err(|_| {
                let _ = std::fs::remove_file(&partial);
            })?;
        records_out.push(json!({
            "material": m.name(), "rounds": table.rounds, "timing": timing,
            "write_seconds": t.elapsed().as_secs_f64(),
            "counts_white_to_move": counts[0], "counts_black_to_move": counts[1],
            "file": path, "payload_sha256": header.payload_sha256,
        }));
        Ok(table)
    };
    // Restart: reuse completed tables whose payload digest verifies.
    let mut solved = Solved::new();
    let mut resumed = Vec::new();
    // Reuse only intact tables written by this exact clean build.
    let entries = if GENERATOR.ends_with("+dirty") {
        Vec::new()
    } else {
        std::fs::read_dir(output)?.collect::<Result<Vec<_>, _>>()?
    };
    for entry in entries {
        let path = entry.path();
        if path.to_string_lossy().ends_with(".gpu.tbp") {
            let read = File::open(&path)
                .map_err(|e| e.to_string())
                .and_then(|f| Table::read(std::io::BufReader::new(f)));
            let (table, header) = match read {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("not resuming {} ({e})", path.display());
                    continue;
                }
            };
            if header.generator != GENERATOR
                || Some(header.material.as_str())
                    != path
                        .file_name()
                        .and_then(|n| n.to_str()?.strip_suffix(".gpu.tbp"))
            {
                eprintln!(
                    "not resuming {} (generator {:?})",
                    path.display(),
                    header.generator
                );
                continue;
            }
            resumed.push(
                json!({"material": header.material, "payload_sha256": header.payload_sha256}),
            );
            solved.insert(header.material, Arc::new(table));
        }
    }
    // solve_tree does not go below a resumed table. Thus resume a table only
    // with all its lower tables, or a bad lower file stays in the output.
    loop {
        let incomplete: Vec<String> = solved
            .iter()
            .filter(|(_, t)| {
                (1..t.material.skeleton_len()).any(|k| {
                    t.material
                        .without(k)
                        .is_ok_and(|l| !solved.contains_key(&l.name()))
                })
            })
            .map(|(name, _)| name.clone())
            .collect();
        if incomplete.is_empty() {
            break;
        }
        for name in incomplete {
            eprintln!("not resuming {name} (a lower table is not resumed)");
            solved.remove(&name);
        }
    }
    resumed.retain(|r| {
        r["material"]
            .as_str()
            .is_some_and(|m| solved.contains_key(m))
    });
    solve_tree(material, &mut solved, &mut solve_one)?;
    Ok(json!({"resumed_verified_tables": resumed, "tables": records_out}))
}
