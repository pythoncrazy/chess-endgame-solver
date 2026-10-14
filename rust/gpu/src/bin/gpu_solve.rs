//! GPU pawnless retrograde solver (cuTile). Same layout and synchronous rounds
//! as `tablebase_core::pawnless`; the CPU reference is used only to compare.
//!
//! One tile lane handles one word: the side to move and all skeleton squares,
//! with the 64 bits indexing the black-king square. Moves are enumerated with
//! fixed loops (piece x direction x step) driven by a small device table, so
//! lanes never branch. Every move reads the child word of the previous round
//! (synchronous Jacobi update), which makes the round number a proof rank.
use cutile::prelude::*;
use serde_json::json;
use std::{fs::File, io::BufWriter, path::PathBuf, sync::Arc, time::Instant};
use tablebase_core::pawnless::{self, Material, Solved, Table, UNRESOLVED, solve_tree};

/// Lanes (words) per tile block.
const BLOCK: usize = 64;
/// Device metadata layout (i64 entries).
const META_PIECES: usize = 8;
const META_PIECE_STRIDE: usize = 8;
const META_DIRECTIONS: usize = 64;
const META_LEN: usize = META_DIRECTIONS + 5 * 8 * 4;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    /// Bit-parallel king neighbourhood.
    fn king_reach(x: Tile<u64, { [64] }>) -> Tile<u64, { [64] }> {
        let not_a: Tile<u64, { [64] }> = constant(0x7f7f7f7f7f7f7f7fu64, shape![64]);
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let not_a = not_a << one;
        let not_h: Tile<u64, { [64] }> = constant(0x7f7f7f7f7f7f7f7fu64, shape![64]);
        let eight: Tile<u64, { [64] }> = constant(8u64, shape![64]);
        let sideways: Tile<u64, { [64] }> = ((x & not_h) << one) | ((x & not_a) >> one);
        let row: Tile<u64, { [64] }> = x | sideways;
        sideways | (row << eight) | (row >> eight)
    }

    /// Gather one i64 metadata entry per lane (uniform or per-lane index).
    unsafe fn meta_at(meta: *const i64, index: Tile<i64, { [64] }>) -> Tile<i64, { [64] }> {
        let base: PointerTile<*const i64, { [] }> = pointer_to_tile(meta);
        let base: PointerTile<*const i64, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const i64, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*const i64, { [64] }> = base.offset_tile(index);
        let loaded: (Tile<i64, { [64] }>, Token) = unsafe {
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

    /// Masked u64 gather; masked lanes read zero.
    unsafe fn gather(
        plane: *const u64,
        index: Tile<i64, { [64] }>,
        mask: Tile<bool, { [64] }>,
    ) -> Tile<u64, { [64] }> {
        let base: PointerTile<*const u64, { [] }> = pointer_to_tile(plane);
        let base: PointerTile<*const u64, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*const u64, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*const u64, { [64] }> = base.offset_tile(index);
        let loaded: (Tile<u64, { [64] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0u64),
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

    /// Lane mask (all ones or zero) from a predicate.
    fn lanes(condition: Tile<bool, { [64] }>) -> Tile<u64, { [64] }> {
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let ones: Tile<u64, { [64] }> = zero - one;
        select(condition, ones, zero)
    }

    fn word_index() -> Tile<i64, { [64] }> {
        let pid = get_tile_block_id();
        let block: Tile<i32, { [64] }> = broadcast_scalar(pid.0, shape![64]);
        let block: Tile<i64, { [64] }> = exti(block);
        let width: Tile<i64, { [64] }> = constant(64i64, shape![64]);
        let lane: Tile<i64, { [64] }> = iota(shape![64]);
        block * width + lane
    }

    /// Static masks: valid positions and positions in check.
    #[cutile::entry()]
    unsafe fn statics(valid_out: *mut u64, check_out: *mut u64, meta: *const i64, m: i32) {
        let w: Tile<i64, { [64] }> = word_index();
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let ones: Tile<u64, { [64] }> = zero - one;
        let zero_i: Tile<i64, { [64] }> = constant(0i64, shape![64]);
        let one_i: Tile<i64, { [64] }> = constant(1i64, shape![64]);
        let six_i: Tile<i64, { [64] }> = constant(6i64, shape![64]);
        let seven_i: Tile<i64, { [64] }> = constant(7i64, shape![64]);
        let eight_i: Tile<i64, { [64] }> = constant(8i64, shape![64]);
        let sixty_three: Tile<i64, { [64] }> = constant(63i64, shape![64]);
        let m_s: Tile<i32, { [64] }> = broadcast_scalar(m, shape![64]);
        let m_t: Tile<i64, { [64] }> = exti(m_s);
        let stm_black: Tile<i64, { [64] }> = (w >> (six_i * m_t)) & one_i;
        // Occupancy and duplicate detection.
        let mut occ: Tile<u64, { [64] }> = zero;
        let mut dup: Tile<u64, { [64] }> = zero;
        for i in 0i32..m {
            let i_s: Tile<i32, { [64] }> = broadcast_scalar(i, shape![64]);
            let i_t: Tile<i64, { [64] }> = exti(i_s);
            let sq: Tile<i64, { [64] }> = (w >> (six_i * (m_t - one_i - i_t))) & sixty_three;
            let bit: Tile<u64, { [64] }> = one << bitcast(sq);
            dup = dup | (occ & bit);
            occ = occ | bit;
        }
        let wk: Tile<i64, { [64] }> = (w >> (six_i * (m_t - one_i))) & sixty_three;
        let mut white_att: Tile<u64, { [64] }> = zero;
        let mut wk_safe: Tile<u64, { [64] }> = ones;
        for i in 0i32..m {
            let i_s: Tile<i32, { [64] }> = broadcast_scalar(i, shape![64]);
            let i_t: Tile<i64, { [64] }> = exti(i_s);
            let sq: Tile<i64, { [64] }> = (w >> (six_i * (m_t - one_i - i_t))) & sixty_three;
            let piece: Tile<i64, { [64] }> = constant(8i64, shape![64]) + i_t * eight_i;
            let role: Tile<i64, { [64] }> = unsafe { meta_at(meta, piece) };
            let white: Tile<i64, { [64] }> = unsafe { meta_at(meta, piece + one_i) };
            let white_l: Tile<u64, { [64] }> = lanes(ne_tile(white, zero_i));
            let file: Tile<i64, { [64] }> = sq & seven_i;
            let rank: Tile<i64, { [64] }> = sq >> constant(3i64, shape![64]);
            for d in 0i32..8i32 {
                let d_s: Tile<i32, { [64] }> = broadcast_scalar(d, shape![64]);
                let d_t: Tile<i64, { [64] }> = exti(d_s);
                let entry: Tile<i64, { [64] }> = constant(64i64, shape![64])
                    + (role * eight_i + d_t) * constant(4i64, shape![64]);
                let df: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry) };
                let dr: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry + one_i) };
                let enabled: Tile<i64, { [64] }> =
                    unsafe { meta_at(meta, entry + constant(2i64, shape![64])) };
                let slides: Tile<i64, { [64] }> =
                    unsafe { meta_at(meta, entry + constant(3i64, shape![64])) };
                let mut alive: Tile<u64, { [64] }> = lanes(ne_tile(enabled, zero_i));
                let mut path: Tile<u64, { [64] }> = zero;
                for k in 1i32..8i32 {
                    let k_s: Tile<i32, { [64] }> = broadcast_scalar(k, shape![64]);
                    let k_t: Tile<i64, { [64] }> = exti(k_s);
                    let first: Tile<u64, { [64] }> = lanes(eq_tile(k_t, one_i));
                    alive = alive & (first | lanes(ne_tile(slides, zero_i)));
                    let tf: Tile<i64, { [64] }> = file + df * k_t;
                    let tr: Tile<i64, { [64] }> = rank + dr * k_t;
                    let on: Tile<u64, { [64] }> = lanes(ge_tile(tf, zero_i))
                        & lanes(lt_tile(tf, eight_i))
                        & lanes(ge_tile(tr, zero_i))
                        & lanes(lt_tile(tr, eight_i));
                    alive = alive & on;
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
        let black: Tile<u64, { [64] }> = lanes(ne_tile(stm_black, zero_i));
        let valid: Tile<u64, { [64] }> =
            (black & base & wk_safe) | ((black ^ ones) & base & (white_att ^ ones));
        let check: Tile<u64, { [64] }> =
            valid & ((black & white_att) | ((black ^ ones) & (wk_safe ^ ones)));
        unsafe { scatter(valid_out, w, valid) };
        unsafe { scatter(check_out, w, check) };
    }

    /// One synchronous round. MODE 0 seeds checkmates (needs `check`); MODE 1
    /// commits wins and losses from the previous round's sets.
    #[cutile::entry()]
    unsafe fn sweep<const MODE: i32>(
        win_out: *mut u64,
        loss_out: *mut u64,
        counter: *mut i64,
        win_in: *const u64,
        loss_in: *const u64,
        valid_in: *const u64,
        check_in: *const u64,
        lower: *const u64,
        meta: *const i64,
        m: i32,
    ) {
        let w: Tile<i64, { [64] }> = word_index();
        let zero: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        let one: Tile<u64, { [64] }> = constant(1u64, shape![64]);
        let ones: Tile<u64, { [64] }> = zero - one;
        let zero_i: Tile<i64, { [64] }> = constant(0i64, shape![64]);
        let one_i: Tile<i64, { [64] }> = constant(1i64, shape![64]);
        let two_i: Tile<i64, { [64] }> = constant(2i64, shape![64]);
        let three_i: Tile<i64, { [64] }> = constant(3i64, shape![64]);
        let four_i: Tile<i64, { [64] }> = constant(4i64, shape![64]);
        let six_i: Tile<i64, { [64] }> = constant(6i64, shape![64]);
        let seven_i: Tile<i64, { [64] }> = constant(7i64, shape![64]);
        let eight_i: Tile<i64, { [64] }> = constant(8i64, shape![64]);
        let sixty_three: Tile<i64, { [64] }> = constant(63i64, shape![64]);
        let all: Tile<bool, { [64] }> = eq_tile(zero_i, zero_i);
        let m_s: Tile<i32, { [64] }> = broadcast_scalar(m, shape![64]);
        let m_t: Tile<i64, { [64] }> = exti(m_s);
        let stm_shift: Tile<i64, { [64] }> = six_i * m_t;
        let stm_bit: Tile<i64, { [64] }> = one_i << stm_shift;
        let lower_stm_bit: Tile<i64, { [64] }> = one_i << (stm_shift - six_i);
        let stm_black: Tile<i64, { [64] }> = (w >> stm_shift) & one_i;
        let black: Tile<u64, { [64] }> = lanes(ne_tile(stm_black, zero_i));
        let mut occ_white: Tile<u64, { [64] }> = zero;
        let mut occ_black: Tile<u64, { [64] }> = zero;
        for i in 0i32..m {
            let i_s: Tile<i32, { [64] }> = broadcast_scalar(i, shape![64]);
            let i_t: Tile<i64, { [64] }> = exti(i_s);
            let sq: Tile<i64, { [64] }> = (w >> (six_i * (m_t - one_i - i_t))) & sixty_three;
            let piece: Tile<i64, { [64] }> = constant(8i64, shape![64]) + i_t * eight_i;
            let white: Tile<i64, { [64] }> = unsafe { meta_at(meta, piece + one_i) };
            let white_l: Tile<u64, { [64] }> = lanes(ne_tile(white, zero_i));
            let bit: Tile<u64, { [64] }> = one << bitcast(sq);
            occ_white = occ_white | (bit & white_l);
            occ_black = occ_black | (bit & (white_l ^ ones));
        }
        let occ: Tile<u64, { [64] }> = occ_white | occ_black;
        let wk: Tile<i64, { [64] }> = (w >> (six_i * (m_t - one_i))) & sixty_three;
        let wk_bit: Tile<u64, { [64] }> = one << bitcast(wk);
        let mut win: Tile<u64, { [64] }> = zero;
        let mut nonwin: Tile<u64, { [64] }> = zero;
        let mut any: Tile<u64, { [64] }> = zero;
        // Skeleton moves of the side to move.
        for i in 0i32..m {
            let i_s: Tile<i32, { [64] }> = broadcast_scalar(i, shape![64]);
            let i_t: Tile<i64, { [64] }> = exti(i_s);
            let shift: Tile<i64, { [64] }> = six_i * (m_t - one_i - i_t);
            let sq: Tile<i64, { [64] }> = (w >> shift) & sixty_three;
            let piece: Tile<i64, { [64] }> = constant(8i64, shape![64]) + i_t * eight_i;
            let role: Tile<i64, { [64] }> = unsafe { meta_at(meta, piece) };
            let white: Tile<i64, { [64] }> = unsafe { meta_at(meta, piece + one_i) };
            let white_l: Tile<u64, { [64] }> = lanes(ne_tile(white, zero_i));
            let mover: Tile<u64, { [64] }> =
                (white_l & (black ^ ones)) | ((white_l ^ ones) & black);
            let own: Tile<u64, { [64] }> = (white_l & occ_white) | ((white_l ^ ones) & occ_black);
            let enemy: Tile<u64, { [64] }> = (occ ^ own) & (wk_bit ^ ones);
            let blockers: Tile<u64, { [64] }> = own | wk_bit;
            let file: Tile<i64, { [64] }> = sq & seven_i;
            let rank: Tile<i64, { [64] }> = sq >> three_i;
            for d in 0i32..8i32 {
                let d_s: Tile<i32, { [64] }> = broadcast_scalar(d, shape![64]);
                let d_t: Tile<i64, { [64] }> = exti(d_s);
                let entry: Tile<i64, { [64] }> =
                    constant(64i64, shape![64]) + (role * eight_i + d_t) * four_i;
                let df: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry) };
                let dr: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry + one_i) };
                let enabled: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry + two_i) };
                let slides: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry + three_i) };
                let mut alive: Tile<u64, { [64] }> = lanes(ne_tile(enabled, zero_i)) & mover;
                let mut path: Tile<u64, { [64] }> = zero;
                for k in 1i32..8i32 {
                    let k_s: Tile<i32, { [64] }> = broadcast_scalar(k, shape![64]);
                    let k_t: Tile<i64, { [64] }> = exti(k_s);
                    let first: Tile<u64, { [64] }> = lanes(eq_tile(k_t, one_i));
                    alive = alive & (first | lanes(ne_tile(slides, zero_i)));
                    let tf: Tile<i64, { [64] }> = file + df * k_t;
                    let tr: Tile<i64, { [64] }> = rank + dr * k_t;
                    let on: Tile<u64, { [64] }> = lanes(ge_tile(tf, zero_i))
                        & lanes(lt_tile(tf, eight_i))
                        & lanes(ge_tile(tr, zero_i))
                        & lanes(lt_tile(tr, eight_i));
                    alive = alive & on;
                    let t: Tile<i64, { [64] }> = (tr * eight_i + tf) & sixty_three;
                    let tbit: Tile<u64, { [64] }> = one << bitcast(t);
                    let hit_own: Tile<u64, { [64] }> = lanes(ne_tile(blockers & tbit, zero));
                    let hit_enemy: Tile<u64, { [64] }> = lanes(ne_tile(enemy & tbit, zero));
                    let moving: Tile<u64, { [64] }> = alive & (hit_own ^ ones);
                    let quiet: Tile<u64, { [64] }> = moving & (hit_enemy ^ ones);
                    let capture: Tile<u64, { [64] }> = moving & hit_enemy;
                    let moved: Tile<i64, { [64] }> = w + ((t - sq) << shift);
                    // Quiet child: same material, other side to move.
                    let child: Tile<i64, { [64] }> = moved ^ stm_bit;
                    let quiet_b: Tile<bool, { [64] }> = ne_tile(quiet, zero);
                    let cw: Tile<u64, { [64] }> = unsafe { gather(win_in, child, quiet_b) };
                    let cl: Tile<u64, { [64] }> = unsafe { gather(loss_in, child, quiet_b) };
                    let cv: Tile<u64, { [64] }> = unsafe { gather(valid_in, child, quiet_b) };
                    // Capture child: remove the captured skeleton piece j.
                    let mut j_t: Tile<i64, { [64] }> = zero_i;
                    for j in 1i32..m {
                        let j_s: Tile<i32, { [64] }> = broadcast_scalar(j, shape![64]);
                        let jj: Tile<i64, { [64] }> = exti(j_s);
                        let sq_j: Tile<i64, { [64] }> =
                            (w >> (six_i * (m_t - one_i - jj))) & sixty_three;
                        j_t = select(eq_tile(sq_j, t), jj, j_t);
                    }
                    let low_shift: Tile<i64, { [64] }> = six_i * (m_t - one_i - j_t);
                    let low_mask: Tile<i64, { [64] }> = (one_i << low_shift) - one_i;
                    let hi: Tile<i64, { [64] }> = moved >> (low_shift + six_i);
                    let low_word: Tile<i64, { [64] }> =
                        ((hi << low_shift) | (moved & low_mask)) ^ lower_stm_bit;
                    let entry_j: Tile<i64, { [64] }> = constant(8i64, shape![64]) + j_t * eight_i;
                    let capture_b: Tile<bool, { [64] }> = ne_tile(capture, zero);
                    let off_v: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry_j + two_i) };
                    let off_w: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry_j + three_i) };
                    let off_l: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry_j + four_i) };
                    let lw: Tile<u64, { [64] }> =
                        unsafe { gather(lower, off_w + low_word, capture_b) };
                    let ll: Tile<u64, { [64] }> =
                        unsafe { gather(lower, off_l + low_word, capture_b) };
                    let lv: Tile<u64, { [64] }> =
                        unsafe { gather(lower, off_v + low_word, capture_b) };
                    let mask: Tile<u64, { [64] }> = ((path | tbit) ^ ones) & (quiet | capture);
                    let ww: Tile<u64, { [64] }> = cw | lw;
                    let ll2: Tile<u64, { [64] }> = cl | ll;
                    let vv: Tile<u64, { [64] }> = cv | lv;
                    win = win | (ll2 & mask);
                    nonwin = nonwin | ((ww ^ ones) & vv & mask);
                    any = any | (vv & mask);
                    path = path | (tbit & alive);
                    let blocked: Tile<u64, { [64] }> = lanes(ne_tile(occ & tbit, zero));
                    alive = alive & (blocked ^ ones);
                }
            }
        }
        // Black king moves inside the word and black king captures.
        let black_b: Tile<bool, { [64] }> = ne_tile(black, zero);
        let child: Tile<i64, { [64] }> = w ^ stm_bit;
        let cw: Tile<u64, { [64] }> = unsafe { gather(win_in, child, black_b) };
        let cl: Tile<u64, { [64] }> = unsafe { gather(loss_in, child, black_b) };
        let cv: Tile<u64, { [64] }> = unsafe { gather(valid_in, child, black_b) };
        win = win | king_reach(cl);
        nonwin = nonwin | king_reach((cw ^ ones) & cv);
        any = any | king_reach(cv);
        for j in 1i32..m {
            let j_s: Tile<i32, { [64] }> = broadcast_scalar(j, shape![64]);
            let jj: Tile<i64, { [64] }> = exti(j_s);
            let entry_j: Tile<i64, { [64] }> = constant(8i64, shape![64]) + jj * eight_i;
            let white: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry_j + one_i) };
            let take: Tile<u64, { [64] }> = lanes(ne_tile(white, zero_i)) & black;
            let take_b: Tile<bool, { [64] }> = ne_tile(take, zero);
            let low_shift: Tile<i64, { [64] }> = six_i * (m_t - one_i - jj);
            let t: Tile<i64, { [64] }> = (w >> low_shift) & sixty_three;
            let low_mask: Tile<i64, { [64] }> = (one_i << low_shift) - one_i;
            let hi: Tile<i64, { [64] }> = w >> (low_shift + six_i);
            let low_word: Tile<i64, { [64] }> =
                ((hi << low_shift) | (w & low_mask)) ^ lower_stm_bit;
            let off_v: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry_j + two_i) };
            let off_w: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry_j + three_i) };
            let off_l: Tile<i64, { [64] }> = unsafe { meta_at(meta, entry_j + four_i) };
            let lw: Tile<u64, { [64] }> = unsafe { gather(lower, off_w + low_word, take_b) };
            let ll: Tile<u64, { [64] }> = unsafe { gather(lower, off_l + low_word, take_b) };
            let lv: Tile<u64, { [64] }> = unsafe { gather(lower, off_v + low_word, take_b) };
            let tu: Tile<u64, { [64] }> = bitcast(t);
            let sources: Tile<u64, { [64] }> = king_reach(one << tu);
            let bit_l: Tile<u64, { [64] }> = lanes(ne_tile((ll >> tu) & one, zero));
            let bit_n: Tile<u64, { [64] }> = lanes(ne_tile((((lw ^ ones) & lv) >> tu) & one, zero));
            let bit_v: Tile<u64, { [64] }> = lanes(ne_tile((lv >> tu) & one, zero));
            win = win | (sources & bit_l);
            nonwin = nonwin | (sources & bit_n);
            any = any | (sources & bit_v);
        }
        let valid: Tile<u64, { [64] }> = unsafe { gather(valid_in, w, all) };
        let old_win: Tile<u64, { [64] }> = unsafe { gather(win_in, w, all) };
        let old_loss: Tile<u64, { [64] }> = unsafe { gather(loss_in, w, all) };
        let mut new_win: Tile<u64, { [64] }> = old_win | (win & valid);
        let mut new_loss: Tile<u64, { [64] }> = old_loss | (valid & any & (nonwin ^ ones));
        if MODE == 0 {
            let check: Tile<u64, { [64] }> = unsafe { gather(check_in, w, all) };
            new_win = zero;
            new_loss = valid & (any ^ ones) & check;
        }
        unsafe { scatter(win_out, w, new_win) };
        unsafe { scatter(loss_out, w, new_loss) };
        let changed: Tile<i64, { [64] }> = select(
            ne_tile((new_win ^ old_win) | (new_loss ^ old_loss), zero),
            one_i,
            zero_i,
        );
        let total: Tile<i64, { [] }> = reduce_sum(changed, 0);
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

    /// Record proof rank `round` for every bit that became a win or a loss.
    /// `rank` holds one u16 per position: `word * 64 + black king square`.
    #[cutile::entry()]
    unsafe fn ranks(
        rank: *mut u16,
        win_old: *const u64,
        loss_old: *const u64,
        win_new: *const u64,
        loss_new: *const u64,
        round: u16,
    ) {
        let w: Tile<i64, { [64] }> = word_index();
        let zero_i: Tile<i64, { [64] }> = constant(0i64, shape![64]);
        let all: Tile<bool, { [64] }> = eq_tile(zero_i, zero_i);
        let wo: Tile<u64, { [64] }> = unsafe { gather(win_old, w, all) };
        let lo: Tile<u64, { [64] }> = unsafe { gather(loss_old, w, all) };
        let wn: Tile<u64, { [64] }> = unsafe { gather(win_new, w, all) };
        let ln: Tile<u64, { [64] }> = unsafe { gather(loss_new, w, all) };
        // Sets are monotone, so XOR yields exactly the new bits.
        let fresh: Tile<u64, { [64] }> = (wn ^ wo) | (ln ^ lo);
        let fresh: Tile<u64, { [64, 64] }> = fresh.reshape(shape![64, 1]).broadcast(shape![64, 64]);
        let column: Tile<u64, { [64] }> = iota(shape![64]);
        let column: Tile<u64, { [64, 64] }> =
            column.reshape(shape![1, 64]).broadcast(shape![64, 64]);
        let one: Tile<u64, { [64, 64] }> = constant(1u64, shape![64, 64]);
        let zero: Tile<u64, { [64, 64] }> = constant(0u64, shape![64, 64]);
        let set: Tile<bool, { [64, 64] }> = ne_tile((fresh >> column) & one, zero);
        let row: Tile<i64, { [64, 64] }> = w.reshape(shape![64, 1]).broadcast(shape![64, 64]);
        let col: Tile<i64, { [64] }> = iota(shape![64]);
        let col: Tile<i64, { [64, 64] }> = col.reshape(shape![1, 64]).broadcast(shape![64, 64]);
        let width: Tile<i64, { [64, 64] }> = constant(64i64, shape![64, 64]);
        let index: Tile<i64, { [64, 64] }> = row * width + col;
        let base: PointerTile<*mut u16, { [] }> = pointer_to_tile(rank);
        let base: PointerTile<*mut u16, { [1, 1] }> = base.reshape(shape![1, 1]);
        let base: PointerTile<*mut u16, { [64, 64] }> = base.broadcast(shape![64, 64]);
        let ptrs: PointerTile<*mut u16, { [64, 64] }> = base.offset_tile(index);
        let value: Tile<u16, { [64, 64] }> = broadcast_scalar(round, shape![64, 64]);
        // SAFETY: index < words * 64, the length of `rank`; only set lanes write.
        unsafe {
            store_ptr_tko(
                ptrs,
                value,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(set),
                None,
                Latency::<0>,
            )
        };
    }
}

fn meta_table(material: &Material, lower_offsets: &[[i64; 3]]) -> Vec<i64> {
    let mut meta = vec![0i64; META_LEN];
    meta[0] = material.skeleton_len() as i64;
    for (i, piece) in material.skeleton.iter().enumerate() {
        let base = META_PIECES + i * META_PIECE_STRIDE;
        meta[base] = role_code(piece.role);
        meta[base + 1] = i64::from(piece.white);
        meta[base + 2..base + 5].copy_from_slice(&lower_offsets[i]);
    }
    for role in [
        pawnless::Role::King,
        pawnless::Role::Queen,
        pawnless::Role::Rook,
        pawnless::Role::Bishop,
        pawnless::Role::Knight,
    ] {
        let (directions, slides) = role.movement();
        for (d, &(df, dr)) in directions.iter().enumerate() {
            let entry = META_DIRECTIONS + (role_code(role) as usize * 8 + d) * 4;
            meta[entry..entry + 4].copy_from_slice(&[
                i64::from(df),
                i64::from(dr),
                1,
                i64::from(slides),
            ]);
        }
    }
    meta
}

fn role_code(role: pawnless::Role) -> i64 {
    match role {
        pawnless::Role::King => 0,
        pawnless::Role::Queen => 1,
        pawnless::Role::Rook => 2,
        pawnless::Role::Bishop => 3,
        pawnless::Role::Knight => 4,
    }
}

type Error = Box<dyn std::error::Error>;

/// Stream-ordered device allocation. cuTile tensors are limited to fewer than
/// 2^31 elements, while six-piece planes hold 2^31 words, so solver buffers use
/// raw driver allocations; the kernels take raw pointers.
struct DeviceVec<T> {
    ptr: cutile::cuda_core::sys::CUdeviceptr,
    len: usize,
    stream: Arc<cutile::cuda_core::Stream>,
    _type: std::marker::PhantomData<T>,
}

impl<T: Copy + Default> DeviceVec<T> {
    fn zeroed(stream: &Arc<cutile::cuda_core::Stream>, len: usize) -> Result<Self, Error> {
        let bytes = len.max(1) * std::mem::size_of::<T>();
        // SAFETY: valid stream; the allocation is zeroed before any use.
        let ptr = unsafe { cutile::cuda_core::malloc_async(bytes, stream)? };
        unsafe { cutile::cuda_core::memset_d8_async(ptr, 0, bytes, stream)? };
        Ok(DeviceVec {
            ptr,
            len,
            stream: stream.clone(),
            _type: std::marker::PhantomData,
        })
    }

    fn filled(
        stream: &Arc<cutile::cuda_core::Stream>,
        byte: u8,
        len: usize,
    ) -> Result<Self, Error> {
        let v = Self::zeroed(stream, len)?;
        // SAFETY: the allocation holds len elements.
        unsafe {
            cutile::cuda_core::memset_d8_async(v.ptr, byte, len * std::mem::size_of::<T>(), stream)?
        };
        Ok(v)
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
        self.len * std::mem::size_of::<T>()
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
    fn zeros<T: Copy + Default>(&self, len: usize) -> Result<DeviceVec<T>, Error> {
        DeviceVec::zeroed(&self.stream, len)
    }

    /// Solve one material on the GPU; returns the table and phase records.
    ///
    /// `ranked`: synchronous double-buffered rounds with device proof ranks.
    /// Otherwise WDL is updated in place (sound because the sets are monotone),
    /// which needs three planes instead of six but yields no proof ranks.
    fn solve(
        &self,
        material: &Material,
        lower: &[Option<&Table>],
        ranked: bool,
    ) -> Result<(Table, serde_json::Value), Error> {
        let start = Instant::now();
        let words = material.words();
        let m = material.skeleton_len() as i32;
        let mut lower_data: Vec<u64> = vec![0];
        let mut offsets = vec![[0i64; 3]; material.skeleton_len()];
        for (k, table) in lower.iter().enumerate() {
            if let Some(t) = table {
                for (slot, plane) in [&t.valid, &t.win, &t.loss].into_iter().enumerate() {
                    offsets[k][slot] = lower_data.len() as i64;
                    lower_data.extend_from_slice(plane);
                }
            } else if k > 0 {
                return Err(format!("missing lower table for skeleton piece {k}").into());
            }
        }
        let meta = DeviceVec::from_host(&self.stream, &meta_table(material, &offsets))?;
        let lower_dev = DeviceVec::from_host(&self.stream, &lower_data)?;
        drop(lower_data);
        let valid = self.zeros::<u64>(words)?;
        let win = [
            self.zeros::<u64>(words)?,
            self.zeros::<u64>(if ranked { words } else { 1 })?,
        ];
        let loss = [
            self.zeros::<u64>(words)?,
            self.zeros::<u64>(if ranked { words } else { 1 })?,
        ];
        // Ranked mode keeps a separate check plane; in place, check bits live
        // in the loss plane until the seed round replaces them by checkmates.
        let check = if ranked {
            Some(self.zeros::<u64>(words)?)
        } else {
            None
        };
        let check_ptr = check.as_ref().unwrap_or(&loss[0]).ptr();
        let rank = if ranked {
            // UNRESOLVED is 0xffff: fill every byte with 0xff.
            Some(DeviceVec::<u16>::filled(&self.stream, 0xff, words * 64)?)
        } else {
            None
        };
        let device_bytes = [&valid, &win[0], &win[1], &loss[0], &loss[1], &lower_dev]
            .iter()
            .map(|v| v.bytes())
            .sum::<usize>()
            + check.as_ref().map_or(0, |c| c.bytes())
            + rank.as_ref().map_or(0, |r| r.bytes());
        let grid = ((words / BLOCK) as u32, 1, 1);
        let device = self.stream.device();
        let (event_start, event_end) = (device.new_event()?, device.new_event()?);
        let prepare_seconds = start.elapsed().as_secs_f64();
        let t0 = Instant::now();
        // SAFETY: every pointer addresses a live device tensor sized for this
        // material; kernels touch words < `words` and lower offsets in range.
        unsafe {
            kernel::statics(valid.ptr(), check_ptr, meta.ptr(), m)
                .grid(grid)
                .sync_on(&self.stream)?;
        }
        let statics_seconds = t0.elapsed().as_secs_f64();
        let mut rounds = 0u32;
        let mut records = Vec::new();
        for round in 0u32.. {
            if round >= u32::from(UNRESOLVED) {
                return Err("proof rank exceeds the 16-bit representation".into());
            }
            let r0 = Instant::now();
            let (src, dst) = if ranked {
                ((round as usize + 1) % 2, round as usize % 2)
            } else {
                (0, 0)
            };
            let counter = self.zeros::<i64>(1)?;
            let mode = if round == 0 { "0" } else { "1" };
            event_start.record(&self.stream)?;
            // SAFETY: as above. Ranked mode reads [src] and writes [dst]; in
            // place both are [0], and monotone updates make races benign.
            unsafe {
                kernel::sweep(
                    win[dst].ptr(),
                    loss[dst].ptr(),
                    counter.ptr(),
                    win[src].ptr(),
                    loss[src].ptr(),
                    valid.ptr(),
                    check_ptr,
                    lower_dev.ptr(),
                    meta.ptr(),
                    m,
                )
                .generics(vec![mode.into()])
                .grid(grid)
                .sync_on(&self.stream)?;
            }
            event_end.record(&self.stream)?;
            event_end.synchronize()?;
            let sweep_event_ms = event_start.elapsed_time(&event_end)?;
            if let Some(rank) = rank.as_ref() {
                // SAFETY: the four planes hold `words` words each.
                unsafe {
                    kernel::ranks(
                        rank.ptr(),
                        win[src].ptr(),
                        loss[src].ptr(),
                        win[dst].ptr(),
                        loss[dst].ptr(),
                        round as u16,
                    )
                    .grid(grid)
                    .sync_on(&self.stream)?;
                }
            }
            let changed = counter.to_host()?[0];
            records.push(json!({
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
        let table = Table {
            material: material.clone(),
            valid: valid.to_host()?,
            win: win[last].to_host()?,
            loss: loss[last].to_host()?,
            rank: match rank {
                Some(r) => r.to_host()?,
                None => Vec::new(),
            },
            rounds,
        };
        let record = json!({
            "mode": if ranked { "synchronous-ranked" } else { "in-place-wdl" },
            "words": words, "device_bytes_allocated": device_bytes,
            "prepare_seconds": prepare_seconds,
            "statics_seconds_including_jit": statics_seconds,
            "rounds_seconds_total": rounds_seconds,
            "download_seconds": t2.elapsed().as_secs_f64(),
            "rounds": records,
        });
        Ok((table, record))
    }
}

fn main() -> Result<(), Error> {
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
    let device = cutile::cuda_core::Device::new(0)?;
    let gpu = Gpu {
        stream: device.new_stream()?,
    };
    let mut records = Vec::new();
    let mut solve_one = |m: &Material, lower: &[Option<&Table>]| -> tablebase_core::Result<Table> {
        let ranked = m.skeleton_len() <= max_ranked;
        let (table, timing) = gpu.solve(m, lower, ranked).map_err(|e| e.to_string())?;
        let path = output.join(format!("{}.gpu.tbp", m.name()));
        let file = File::create(&path).map_err(|e| e.to_string())?;
        let header = table.write(BufWriter::new(file))?;
        records.push(json!({
            "material": m.name(), "rounds": table.rounds, "timing": timing,
            "counts_white_to_move": table.counts()[0], "counts_black_to_move": table.counts()[1],
            "file": path, "payload_sha256": header.payload_sha256,
        }));
        Ok(table)
    };
    solve_tree(&material, &mut Solved::new(), &mut solve_one)?;
    println!(
        "{}",
        json!({"solver": "gpu-cutile", "device": std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(), "tables": records})
    );
    Ok(())
}
