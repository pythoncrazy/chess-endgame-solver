//! GPU pawnless retrograde solver (cuTile). Same layout and synchronous rounds
//! as `tablebase_core::pawnless`; the CPU reference is used only to compare.
//!
//! One tile lane handles one word: the side to move and all skeleton squares,
//! with the 64 bits indexing the black-king square. Device state is one record
//! per word, `[win | illegal, loss | illegal]` (both set means illegal), so a
//! child's value is a single 16-byte pair read. A 64-word block shares the side
//! to move, so pieces of the other side are skipped with a uniform branch and
//! each role walks only its own directions and steps. Synchronous rounds read
//! the previous round only, which makes the round number a proof rank.
use cutile::jit_cache::{self, FileSystemJitStore};
use cutile::prelude::*;
use serde_json::json;
use std::{fs::File, io::BufWriter, path::PathBuf, sync::Arc, time::Instant};
use tablebase_core::pawnless::{self, Material, Solved, Table, UNRESOLVED, solve_tree};

/// Program and git revision recorded in table headers (see build.rs).
const GENERATOR: &str = concat!("gpu_solve git:", env!("TB_SOURCE_REVISION"));
/// Lanes (words) per tile block.
const BLOCK: usize = 64;
/// i32 metadata: m at 0; piece i at 8 + 4i (role, white, directions, slides);
/// direction d of role r at 64 + 2(8r + d) (file step, rank step).
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

    fn i64s(value: i32) -> Tile<i64, { [64] }> {
        let t: Tile<i32, { [64] }> = broadcast_scalar(value, shape![64]);
        exti(t)
    }

    /// Scalar block index of the 2-D grid (block ids above 2^24 on one axis
    /// are unreliable in the pinned cuTile, so large launches use grid Y).
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

    /// Masked gather of records `[win | illegal, loss | illegal]`; masked lanes
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
        // cannot be broadcast: masked-off lanes read record 0 (always valid)
        // and are zeroed afterwards with the widened u64 lane mask.
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

    /// Initial records (illegal positions marked as both win and loss) and
    /// the positions in which the side to move is in check.
    #[cutile::entry()]
    unsafe fn statics(records: *mut u64, check_out: *mut u64, meta: *const i32) {
        // Scalars come from device memory: scalar kernel arguments enter the
        // pinned cuTile's in-memory kernel key and force front-end reruns.
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

    /// One round. MODE 0 seeds checkmates; MODE 1 commits wins and losses
    /// derived from `input` into `output` (the same buffer when in place).
    #[cutile::entry()]
    unsafe fn sweep<const MODE: i32, const RANKED: i32>(
        output: *mut u64,
        counter: *mut i64,
        rank: *mut u16,
        input: *const u64,
        check_in: *const u64,
        lower: *const u64,
        meta: *const i32,
        round: *const u16,
    ) {
        let m: i32 = unsafe { scalar_at(meta, 0i32) };
        let blk: i32 = block_id();
        let w: Tile<i64, { [64] }> = word_index(blk);
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
        let m_t: Tile<i64, { [64] }> = i64s(m);
        let stm_shift: Tile<i64, { [64] }> = six_i * m_t;
        let stm_bit: Tile<i64, { [64] }> = one_i << stm_shift;
        let lower_stm_bit: Tile<i64, { [64] }> = one_i << (stm_shift - six_i);
        // Records per lower table: 2 * 64^(m - 1).
        let lower_words: Tile<i64, { [64] }> = one_i << (stm_shift - six_i + one_i);
        let black_to_move: i32 = (blk >> (6i32 * m - 6i32)) & 1i32;
        let mover_white: i32 = 1i32 - black_to_move;
        let mut occ_white: Tile<u64, { [64] }> = zero;
        let mut occ_black: Tile<u64, { [64] }> = zero;
        for i in 0i32..m {
            let white: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * i) };
            let white_l: Tile<u64, { [64] }> = lanes(ne_tile(i64s(white), zero_i));
            let sq: Tile<i64, { [64] }> = (w >> i64s(6i32 * (m - 1i32 - i))) & sixty_three;
            let bit: Tile<u64, { [64] }> = one << bitcast(sq);
            occ_white = occ_white | (bit & white_l);
            occ_black = occ_black | (bit & (white_l ^ ones));
        }
        let occ: Tile<u64, { [64] }> = occ_white | occ_black;
        let own: Tile<u64, { [64] }> = if mover_white == 1i32 {
            occ_white
        } else {
            occ_black
        };
        let wk: Tile<i64, { [64] }> = (w >> i64s(6i32 * (m - 1i32))) & sixty_three;
        let wk_bit: Tile<u64, { [64] }> = one << bitcast(wk);
        let enemy: Tile<u64, { [64] }> = (occ ^ own) & (wk_bit ^ ones);
        let blockers: Tile<u64, { [64] }> = own | wk_bit;
        let mut win: Tile<u64, { [64] }> = zero;
        let mut nonwin: Tile<u64, { [64] }> = zero;
        let mut any: Tile<u64, { [64] }> = zero;
        // Skeleton moves: only the side to move's pieces, each role's own rays.
        for i in 0i32..m {
            let white: i32 = unsafe { scalar_at(meta, 9i32 + 4i32 * i) };
            if white == mover_white {
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
                        let tbit: Tile<u64, { [64] }> = one << bitcast(t);
                        let hit_own: Tile<u64, { [64] }> = lanes(ne_tile(blockers & tbit, zero));
                        let hit_enemy: Tile<u64, { [64] }> = lanes(ne_tile(enemy & tbit, zero));
                        let moving: Tile<u64, { [64] }> = alive & (hit_own ^ ones);
                        let quiet: Tile<u64, { [64] }> = moving & (hit_enemy ^ ones);
                        let capture: Tile<u64, { [64] }> = moving & hit_enemy;
                        let moved: Tile<i64, { [64] }> = w + ((t - sq) << shift);
                        let qp: Tile<u64, { [64, 2] }> =
                            unsafe { gather_pair(input, moved ^ stm_bit, quiet) };
                        // Captured skeleton piece j: its lower table is the record
                        // range (j - 1) * lower_words, indexed without piece j.
                        let mut j_t: Tile<i64, { [64] }> = zero_i;
                        for j in 1i32..m {
                            let jj: Tile<i64, { [64] }> = i64s(j);
                            let sq_j: Tile<i64, { [64] }> =
                                (w >> i64s(6i32 * (m - 1i32 - j))) & sixty_three;
                            j_t = select(eq_tile(sq_j, t), jj, j_t);
                        }
                        let low_shift: Tile<i64, { [64] }> = six_i * (m_t - one_i - j_t);
                        let low_mask: Tile<i64, { [64] }> = (one_i << low_shift) - one_i;
                        let hi: Tile<i64, { [64] }> = moved >> (low_shift + six_i);
                        let low_word: Tile<i64, { [64] }> =
                            ((hi << low_shift) | (moved & low_mask)) ^ lower_stm_bit;
                        let lower_index: Tile<i64, { [64] }> =
                            (j_t - one_i) * lower_words + low_word;
                        let lp: Tile<u64, { [64, 2] }> =
                            unsafe { gather_pair(lower, lower_index, capture) };
                        let cw: Tile<u64, { [64] }> = pair_part(qp, 0i64) | pair_part(lp, 0i64);
                        let cl: Tile<u64, { [64] }> = pair_part(qp, 1i64) | pair_part(lp, 1i64);
                        let mask: Tile<u64, { [64] }> = ((path | tbit) ^ ones) & (quiet | capture);
                        win = win | (cl & (cw ^ ones) & mask);
                        nonwin = nonwin | ((cw ^ ones) & mask);
                        any = any | (((cw & cl) ^ ones) & mask);
                        path = path | (tbit & alive);
                        alive = alive & (lanes(ne_tile(occ & tbit, zero)) ^ ones);
                    }
                }
            }
        }
        // Black king moves inside the word, and black king captures.
        if black_to_move == 1i32 {
            let cp: Tile<u64, { [64, 2] }> = unsafe { gather_pair(input, w ^ stm_bit, ones) };
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
                    let lower_index: Tile<i64, { [64] }> = i64s(j - 1i32) * lower_words + low_word;
                    let lp: Tile<u64, { [64, 2] }> =
                        unsafe { gather_pair(lower, lower_index, ones) };
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
        let op: Tile<u64, { [64, 2] }> = unsafe { gather_pair(input, w, ones) };
        let old_win: Tile<u64, { [64] }> = pair_part(op, 0i64);
        let old_loss: Tile<u64, { [64] }> = pair_part(op, 1i64);
        let valid: Tile<u64, { [64] }> = (old_win & old_loss) ^ ones;
        let mut new_win: Tile<u64, { [64] }> = old_win | (win & valid);
        let mut new_loss: Tile<u64, { [64] }> = old_loss | (valid & any & (nonwin ^ ones));
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
        if RANKED == 1 {
            if block_changed > 0i64 {
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
                let ptrs: PointerTile<*mut u16, { [64, 64] }> = base.offset_tile(row * width + col);
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

/// Largest grid X used; ids above 2^24 on one axis are unreliable in the pinned
/// cuTile, so larger launches add a Y dimension. Word counts are powers of two.
const GRID_X: usize = 1 << 20;

fn launch_grid(words: usize) -> (u32, u32, u32) {
    let blocks = words / BLOCK;
    if blocks <= GRID_X {
        (blocks as u32, 1, 1)
    } else {
        (GRID_X as u32, (blocks / GRID_X) as u32, 1)
    }
}

/// Stream-ordered device allocation. cuTile tensors are limited to fewer than
/// 2^31 elements, while six-piece state holds more, so solver buffers use raw
/// driver allocations; the kernels take raw pointers.
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
        // SAFETY: valid stream; the allocation is initialized before any use.
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
        // SAFETY: data and the allocation both hold data.len() elements; the
        // stream is synchronized before `data` can be dropped by the caller.
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
        // SAFETY: both hold len elements; synchronized before `data` is released.
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
        // SAFETY: both buffers hold len elements; synchronized before return.
        unsafe {
            cutile::cuda_core::memcpy_dtoh_async(
                out.as_mut_ptr(),
                self.ptr,
                self.len,
                &self.stream,
            )?;
            self.stream.synchronize()?;
        }
        Ok(out)
    }

    fn ptr(&self) -> cutile::tile_kernel::DevicePointer<T> {
        // SAFETY: `ptr` is a live allocation of `len` values of T.
        unsafe { cutile::tile_kernel::DevicePointer::from_cu_deviceptr(self.ptr) }
    }

    fn bytes(&self) -> usize {
        self.len.max(1) * std::mem::size_of::<T>()
    }
}

impl<T> Drop for DeviceVec<T> {
    fn drop(&mut self) {
        // SAFETY: allocated by malloc_async on this stream; stream order puts the
        // free after every kernel that used it.
        let _ = unsafe { cutile::cuda_core::free_async(self.ptr, &self.stream) };
    }
}

struct Gpu {
    stream: Arc<cutile::cuda_core::Stream>,
}

impl Gpu {
    /// Solve one material on the GPU; returns the table and phase records.
    ///
    /// `ranked`: synchronous double-buffered rounds with device proof ranks.
    /// Otherwise WDL is updated in place (sound because the sets are monotone),
    /// which halves the state but yields no proof ranks.
    fn solve(
        &self,
        material: &Material,
        lower: &[Option<&Table>],
        ranked: bool,
    ) -> Result<(Table, serde_json::Value), Error> {
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
        let device_bytes =
            state[0].bytes() + state[1].bytes() + check.bytes() + lower_dev.bytes() + rank.bytes();
        let grid = launch_grid(words);
        let device = self.stream.device();
        let (event_start, event_end) = (device.new_event()?, device.new_event()?);
        let prepare_seconds = start.elapsed().as_secs_f64();
        let t0 = Instant::now();
        // SAFETY: every pointer addresses a live device buffer sized for this
        // material; kernels touch words < `words` and lower records in range.
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
        let mut rounds = 0u32;
        let mut records_log = Vec::new();
        for round in 0u32.. {
            if round >= u32::from(UNRESOLVED) {
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
            event_start.record(&self.stream)?;
            // SAFETY: as above. In place, input and output are the same buffer;
            // monotone updates make concurrent reads of old or new values sound.
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
                )
                .generics(vec![mode.into(), if ranked { "1" } else { "0" }.into()])
                .grid(grid)
                .sync_on(&self.stream)?;
            }
            event_end.record(&self.stream)?;
            event_end.synchronize()?;
            let sweep_event_ms = event_start.elapsed_time(&event_end)?;
            let changed = counter.to_host()?[0];
            records_log.push(json!({
                "round": round, "changed_words": changed,
                "host_seconds": r0.elapsed().as_secs_f64(), "sweep_event_ms": sweep_event_ms,
            }));
            if round > 0 && changed == 0 {
                break;
            }
            rounds = round;
        }
        let rounds_seconds = t0.elapsed().as_secs_f64() - statics_seconds;
        let t2 = Instant::now();
        let last = if ranked { rounds as usize % 2 } else { 0 };
        let wl = state[last].to_host()?;
        let mut table = Table {
            material: material.clone(),
            valid: vec![0; words],
            win: vec![0; words],
            loss: vec![0; words],
            rank: if ranked { rank.to_host()? } else { Vec::new() },
            rounds,
        };
        for w in 0..words {
            let valid = !(wl[2 * w] & wl[2 * w + 1]);
            table.valid[w] = valid;
            table.win[w] = wl[2 * w] & valid;
            table.loss[w] = wl[2 * w + 1] & valid;
        }
        let record = json!({
            "mode": if ranked { "synchronous-ranked" } else { "in-place-wdl" },
            "words": words, "device_bytes_allocated": device_bytes,
            "prepare_seconds": prepare_seconds,
            "statics_seconds_including_jit": statics_seconds,
            "rounds_seconds_total": rounds_seconds,
            "download_seconds": t2.elapsed().as_secs_f64(),
            "rounds": records_log,
        });
        Ok((table, record))
    }
}

fn main() -> Result<(), Error> {
    let wall = Instant::now();
    let mut args = std::env::args().skip(1);
    let usage = "usage: gpu_solve MATERIAL OUTPUT_DIRECTORY [MAX_RANKED_SKELETON]";
    let material = Material::parse(&args.next().ok_or(usage)?)?;
    let output = PathBuf::from(args.next().ok_or(usage)?);
    // Proof ranks need 128 bytes per word: keep them up to five pieces.
    let max_ranked: usize = match args.next() {
        Some(v) => v.parse()?,
        None => 4,
    };
    std::fs::create_dir_all(&output)?;
    // Compiled kernels persist across runs (keyed by source, generics, toolkit).
    jit_cache::enable(Arc::new(FileSystemJitStore::default_location()?));
    let device = cutile::cuda_core::Device::new(0)?;
    let gpu = Gpu {
        stream: device.new_stream()?,
    };
    let mut records_out = Vec::new();
    let mut solve_one = |m: &Material, lower: &[Option<&Table>]| -> tablebase_core::Result<Table> {
        let ranked = m.skeleton_len() <= max_ranked;
        let (table, timing) = gpu.solve(m, lower, ranked).map_err(|e| e.to_string())?;
        let t = Instant::now();
        let path = output.join(format!("{}.gpu.tbp", m.name()));
        let file = File::create(&path).map_err(|e| e.to_string())?;
        let header = table.write(BufWriter::new(file), GENERATOR)?;
        records_out.push(json!({
            "material": m.name(), "rounds": table.rounds, "timing": timing,
            "write_seconds": t.elapsed().as_secs_f64(),
            "counts_white_to_move": table.counts()[0], "counts_black_to_move": table.counts()[1],
            "file": path, "payload_sha256": header.payload_sha256,
        }));
        Ok(table)
    };
    // Restart: reuse completed tables whose payload digest verifies.
    let mut solved = Solved::new();
    let mut resumed = Vec::new();
    for entry in std::fs::read_dir(&output)? {
        let path = entry?.path();
        if path.to_string_lossy().ends_with(".gpu.tbp") {
            let (table, header) = Table::read(std::io::BufReader::new(File::open(&path)?))?;
            // Reuse only intact tables written by this exact clean build.
            if GENERATOR.ends_with("+dirty") || header.generator != GENERATOR {
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
    solve_tree(&material, &mut solved, &mut solve_one)?;
    println!(
        "{}",
        json!({
            "solver": "gpu-cutile", "generator": GENERATOR,
            "device": std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),
            "wall_seconds": wall.elapsed().as_secs_f64(),
            "resumed_verified_tables": resumed, "tables": records_out,
        })
    );
    Ok(())
}
