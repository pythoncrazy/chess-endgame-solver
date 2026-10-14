//! Probe: do cuTile pointer gathers/scatters address beyond 2^31 bytes?
//! Writes lane indices at large i64 element offsets, then reads them back.
use cutile::prelude::*;
use std::sync::Arc;

#[cutile::module]
mod kernel {
    use cutile::core::*;

    /// Store each word's own index at pid * 64 + lane (index derived from the
    /// i32 block id, as in the solver).
    #[cutile::entry()]
    unsafe fn store_own_index(buffer: *mut u64) {
        // 2-D grid: block = pid.1 * grid.0 + pid.0, in i64.
        let pid = get_tile_block_id();
        let grid = get_num_tile_blocks();
        let x: Tile<i32, { [64] }> = broadcast_scalar(pid.0, shape![64]);
        let y: Tile<i32, { [64] }> = broadcast_scalar(pid.1, shape![64]);
        let gx: Tile<i32, { [64] }> = broadcast_scalar(grid.0, shape![64]);
        let x: Tile<i64, { [64] }> = exti(x);
        let y: Tile<i64, { [64] }> = exti(y);
        let gx: Tile<i64, { [64] }> = exti(gx);
        let block: Tile<i64, { [64] }> = y * gx + x;
        let width: Tile<i64, { [64] }> = constant(64i64, shape![64]);
        let lane: Tile<i64, { [64] }> = iota(shape![64]);
        let w: Tile<i64, { [64] }> = block * width + lane;
        let base: PointerTile<*mut u64, { [] }> = pointer_to_tile(buffer);
        let base: PointerTile<*mut u64, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut u64, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*mut u64, { [64] }> = base.offset_tile(w);
        let value: Tile<u64, { [64] }> = bitcast(w);
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

    #[cutile::entry()]
    unsafe fn roundtrip(buffer: *mut u64, out: *mut u64, start: i64, stride: i64) {
        let lane: Tile<i64, { [64] }> = iota(shape![64]);
        let start_t: Tile<i64, { [64] }> = broadcast_scalar(start, shape![64]);
        let stride_t: Tile<i64, { [64] }> = broadcast_scalar(stride, shape![64]);
        let index: Tile<i64, { [64] }> = start_t + lane * stride_t;
        let value: Tile<u64, { [64] }> = bitcast(index);
        let base: PointerTile<*mut u64, { [] }> = pointer_to_tile(buffer);
        let base: PointerTile<*mut u64, { [1] }> = base.reshape(shape![1]);
        let base: PointerTile<*mut u64, { [64] }> = base.broadcast(shape![64]);
        let ptrs: PointerTile<*mut u64, { [64] }> = base.offset_tile(index);
        let token: Token = unsafe {
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
        let loaded: (Tile<u64, { [64] }>, Token) = unsafe {
            load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                Some(token),
                Latency::<0>,
            )
        };
        let lane_ptr: PointerTile<*mut u64, { [] }> = pointer_to_tile(out);
        let lane_ptr: PointerTile<*mut u64, { [1] }> = lane_ptr.reshape(shape![1]);
        let lane_ptr: PointerTile<*mut u64, { [64] }> = lane_ptr.broadcast(shape![64]);
        let out_ptrs: PointerTile<*mut u64, { [64] }> = lane_ptr.offset_tile(lane);
        unsafe {
            store_ptr_tko(
                out_ptrs,
                loaded.0,
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
    let device = cutile::cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;
    let elements: usize = (1usize << 31) + (1 << 20);
    let bytes = elements * 8;
    // SAFETY: fresh allocation on a valid stream, freed at the end.
    let ptr = unsafe { cutile::cuda_core::malloc_async(bytes, &stream)? };
    let out = api::zeros::<u64>(&[64]).sync_on(&stream)?;
    let stride = (elements as i64 - 1) / 63;
    for (start, step) in [
        (0i64, 1i64),
        (0, stride),
        ((1 << 28) - 32, 1),
        ((1 << 31) - 32, 1),
    ] {
        // SAFETY: indices start + lane * step < elements by construction.
        unsafe {
            kernel::roundtrip(
                cutile::tile_kernel::DevicePointer::from_cu_deviceptr(ptr),
                out.device_pointer(),
                start,
                step,
            )
            .grid((1, 1, 1))
            .sync_on(&stream)?;
        }
        let got = out.dup().to_host_vec().sync_on(&stream)?;
        let ok = got
            .iter()
            .enumerate()
            .all(|(lane, &v)| v == (start + lane as i64 * step) as u64);
        println!(
            "start={start} step={step} max_index={} ok={ok}",
            start + 63 * step
        );
    }
    // Block-id-derived indices over 2^31 words (2^25 blocks of 64).
    let words: usize = 1 << 31;
    // SAFETY: the buffer holds `elements` >= words u64 values.
    unsafe {
        cutile::cuda_core::memset_d8_async(ptr, 0xee, bytes, &stream)?;
        kernel::store_own_index(cutile::tile_kernel::DevicePointer::from_cu_deviceptr(ptr))
            .grid((1 << 20, ((words / 64) >> 20) as u32, 1))
            .sync_on(&stream)?;
    }
    // Binary search for the first index not holding its own value.
    let read = |index: usize| -> Result<u64, Box<dyn std::error::Error>> {
        let mut value = 0u64;
        // SAFETY: index < elements; one u64 copied and synchronized.
        unsafe {
            cutile::cuda_core::memcpy_dtoh_async(&mut value, ptr + (index * 8) as u64, 1, &stream)?;
            stream.synchronize()?;
        }
        Ok(value)
    };
    let (mut lo, mut hi) = (0usize, words - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if read(mid)? == mid as u64 {
            lo = mid
        } else {
            hi = mid
        }
    }
    println!(
        "last verified index {lo}, first failing {hi} ({hi:#x}), value there {:#x}",
        read(hi)?
    );
    for index in [
        0usize,
        5,
        (1 << 28) - 1,
        1 << 28,
        (1 << 29) + 7,
        (1 << 30) + 3,
        words - 1,
    ] {
        let mut value = 0u64;
        // SAFETY: index < elements; one u64 copied and synchronized.
        unsafe {
            cutile::cuda_core::memcpy_dtoh_async(&mut value, ptr + (index * 8) as u64, 1, &stream)?;
            stream.synchronize()?;
        }
        println!(
            "pid-derived index {index}: stored {value:#x} ok={}",
            value == index as u64
        );
    }
    unsafe { cutile::cuda_core::free_async(ptr, &Arc::clone(&stream))? };
    Ok(())
}
