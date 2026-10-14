//! Feature probe for the pinned cuTile JIT: loop-carried tiles, const-generic
//! branches, per-lane i64 index arithmetic and masked u64 pointer gathers.
use cutile::prelude::*;
use std::sync::Arc;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    unsafe fn probe<const MODE: i32>(out_ptr: *mut u64, in_ptr: *const u64, len: i64) {
        let pid = get_tile_block_id();
        let block: Tile<i32, { [128] }> = broadcast_scalar(pid.0, shape![128]);
        let block: Tile<i64, { [128] }> = exti(block);
        let width: Tile<i64, { [128] }> = constant(128i64, shape![128]);
        let lane: Tile<i64, { [128] }> = iota(shape![128]);
        let idx: Tile<i64, { [128] }> = block * width + lane;
        let len_t: Tile<i64, { [128] }> = broadcast_scalar(len, shape![128]);
        let in_ptrs0: PointerTile<*const u64, { [] }> = pointer_to_tile(in_ptr);
        let in_ptrs1: PointerTile<*const u64, { [1] }> = in_ptrs0.reshape(shape![1]);
        let in_ptrs: PointerTile<*const u64, { [128] }> = in_ptrs1.broadcast(shape![128]);
        let mut acc: Tile<u64, { [128] }> = constant(0u64, shape![128]);
        for step in 0i32..4i32 {
            // Gather in[(idx + step * 7) % len], masked to even lanes.
            let s7: Tile<i32, { [128] }> = broadcast_scalar(step * 7i32, shape![128]);
            let s7: Tile<i64, { [128] }> = exti(s7);
            let shifted: Tile<i64, { [128] }> = idx + s7;
            let src: Tile<i64, { [128] }> = shifted % len_t;
            let two: Tile<i64, { [128] }> = constant(2i64, shape![128]);
            let zero: Tile<i64, { [128] }> = constant(0i64, shape![128]);
            let parity: Tile<i64, { [128] }> = idx % two;
            let even: Tile<bool, { [128] }> = eq_tile(parity, zero);
            // SAFETY: src is reduced modulo len, so every unmasked lane is in bounds.
            let src_ptrs: PointerTile<*const u64, { [128] }> = in_ptrs.offset_tile(src);
            let loaded: (Tile<u64, { [128] }>, Token) = unsafe {
                load_ptr_tko(
                    src_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(even),
                    Some(0u64),
                    None,
                    Latency::<0>,
                )
            };
            let v: Tile<u64, { [128] }> = loaded.0;
            acc = acc | v;
        }
        if MODE == 1 {
            let flip: Tile<u64, { [128] }> = constant(0x0101010101010101u64, shape![128]);
            acc = acc ^ flip;
        }
        let out_ptrs0: PointerTile<*mut u64, { [] }> = pointer_to_tile(out_ptr);
        let out_ptrs1: PointerTile<*mut u64, { [1] }> = out_ptrs0.reshape(shape![1]);
        let out_ptrs: PointerTile<*mut u64, { [128] }> = out_ptrs1.broadcast(shape![128]);
        let in_range: Tile<bool, { [128] }> = lt_tile(idx, len_t);
        let dst_ptrs: PointerTile<*mut u64, { [128] }> = out_ptrs.offset_tile(idx);
        // SAFETY: stores are masked to idx < len.
        unsafe {
            store_ptr_tko(
                dst_ptrs,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(in_range),
                None,
                Latency::<0>,
            )
        };
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let len = 1000usize;
    let host: Vec<u64> = (0..len as u64)
        .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    let device = cutile::cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;
    let input = api::copy_host_vec_to_device(&Arc::new(host.clone())).sync_on(&stream)?;
    for mode in [0i32, 1] {
        let output = api::zeros::<u64>(&[len]).sync_on(&stream)?;
        // SAFETY: pointers address live device tensors of `len` elements; all
        // gathers are reduced modulo len and stores are masked to idx < len.
        unsafe {
            kernel::probe(output.device_pointer(), input.device_pointer(), len as i64)
                .generics(vec![mode.to_string()])
                .grid((len.div_ceil(128) as u32, 1, 1))
                .sync_on(&stream)?;
        }
        let got = output.to_host_vec().sync_on(&stream)?;
        let expected: Vec<u64> = (0..len)
            .map(|i| {
                let mut acc = 0;
                if i % 2 == 0 {
                    for step in 0..4 {
                        acc |= host[(i + step * 7) % len];
                    }
                }
                if mode == 1 {
                    acc ^= 0x0101010101010101;
                }
                acc
            })
            .collect();
        let bad = (0..len).filter(|&i| got[i] != expected[i]).count();
        println!("mode {mode}: {bad} mismatches of {len}");
        if bad != 0 {
            return Err("tile probe mismatch".into());
        }
    }
    Ok(())
}
