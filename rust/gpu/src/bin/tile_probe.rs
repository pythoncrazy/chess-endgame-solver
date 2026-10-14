//! Feature probe for the pinned cuTile JIT: loop-carried tiles, scalar loads
//! from a device table, runtime scalar branches and loop bounds, i64 lane
//! arithmetic, and [64, 2] pair gathers.
use cutile::prelude::*;
use std::sync::Arc;

#[cutile::module]
mod kernel {
    use cutile::core::*;

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

    /// out[w] = table[pid % 4] * (pair[w].0 + pair[w].1 when table[4] == 1).
    #[cutile::entry()]
    unsafe fn probe(out_ptr: *mut u64, pairs: *const u64, table: *const i32) {
        let pid = get_tile_block_id();
        let block: Tile<i32, { [64] }> = broadcast_scalar(pid.0, shape![64]);
        let block: Tile<i64, { [64] }> = exti(block);
        let width: Tile<i64, { [64] }> = constant(64i64, shape![64]);
        let lane: Tile<i64, { [64] }> = iota(shape![64]);
        let w: Tile<i64, { [64] }> = block * width + lane;
        let two: Tile<i64, { [64] }> = constant(2i64, shape![64]);
        let doubled: Tile<i64, { [64] }> = w * two;
        let first: Tile<i64, { [64, 2] }> = doubled.reshape(shape![64, 1]).broadcast(shape![64, 2]);
        let col: Tile<i64, { [2] }> = iota(shape![2]);
        let col: Tile<i64, { [64, 2] }> = col.reshape(shape![1, 2]).broadcast(shape![64, 2]);
        let idx: Tile<i64, { [64, 2] }> = first + col;
        let base: PointerTile<*const u64, { [] }> = pointer_to_tile(pairs);
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
        let zero2: Tile<u64, { [64, 2] }> = constant(0u64, shape![64, 2]);
        let zero_c: Tile<i64, { [64, 2] }> = constant(0i64, shape![64, 2]);
        let one_c: Tile<i64, { [64, 2] }> = constant(1i64, shape![64, 2]);
        let pair: Tile<u64, { [64, 2] }> = loaded.0;
        let a: Tile<u64, { [64] }> = reduce_sum(select(eq_tile(col, zero_c), pair, zero2), 1);
        let b: Tile<u64, { [64] }> = reduce_sum(select(eq_tile(col, one_c), pair, zero2), 1);
        let count: i32 = unsafe { scalar_at(table, pid.0 % 4i32) };
        let use_b: i32 = unsafe { scalar_at(table, 4i32) };
        let mut acc: Tile<u64, { [64] }> = constant(0u64, shape![64]);
        for _k in 0i32..count {
            acc = acc + a;
            if use_b == 1i32 {
                acc = acc + b;
            }
        }
        let out: PointerTile<*mut u64, { [] }> = pointer_to_tile(out_ptr);
        let out: PointerTile<*mut u64, { [1] }> = out.reshape(shape![1]);
        let out: PointerTile<*mut u64, { [64] }> = out.broadcast(shape![64]);
        let dst: PointerTile<*mut u64, { [64] }> = out.offset_tile(w);
        unsafe {
            store_ptr_tko(
                dst,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Latency::<0>,
            )
        };
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words = 64 * 8;
    let pairs: Vec<u64> = (0..2 * words as u64).map(|i| i * 3 + 1).collect();
    let table: Vec<i32> = vec![1, 3, 0, 7, 1];
    let device = cutile::cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;
    let pairs_dev = api::copy_host_vec_to_device(&Arc::new(pairs.clone())).sync_on(&stream)?;
    let table_dev = api::copy_host_vec_to_device(&Arc::new(table.clone())).sync_on(&stream)?;
    let out = api::zeros::<u64>(&[words]).sync_on(&stream)?;
    // SAFETY: pairs hold 2 * words values, out holds words, the table 5 entries.
    unsafe {
        kernel::probe(
            out.device_pointer(),
            pairs_dev.device_pointer(),
            table_dev.device_pointer(),
        )
        .grid(((words / 64) as u32, 1, 1))
        .sync_on(&stream)?;
    }
    let got = out.to_host_vec().sync_on(&stream)?;
    let bad = (0..words)
        .filter(|&w| {
            let count = table[(w / 64) % 4] as u64;
            got[w] != count * (pairs[2 * w] + pairs[2 * w + 1])
        })
        .count();
    println!("{bad} mismatches of {words}");
    if bad != 0 {
        return Err("tile probe mismatch".into());
    }
    Ok(())
}
