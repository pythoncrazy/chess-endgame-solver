//! Shared-geometry INT8/FP16/FP8 tensor cores vs packed knight stencils.
//! CPU builds legality masks; this measures resident operators, not generation.
use cutile::prelude::*;
use serde_json::json;
use std::{sync::Arc, time::Instant};
use tablebase_core::fiber::{knight_reach, stencil};

#[cutile::module]
mod kernel {
    use cutile::core::*;

    #[cutile::entry()]
    fn knight_tc<E: ElementType, A: ElementType, const BN: i32>(
        out: &mut Tensor<u64, { [BN, 2] }>,
        geometry: &Tensor<E, { [64, 64] }>,
        wins: &Tensor<u64, { [-1] }>,
        losses: &Tensor<u64, { [-1] }>,
        destinations: &Tensor<u64, { [-1] }>,
        sources: &Tensor<u64, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let w = wins
            .partition(shape![BN])
            .load([pid.0])
            .reshape(shape![1, BN])
            .broadcast(shape![64, BN]);
        let l = losses
            .partition(shape![BN])
            .load([pid.0])
            .reshape(shape![1, BN])
            .broadcast(shape![64, BN]);
        let d = destinations
            .partition(shape![BN])
            .load([pid.0])
            .reshape(shape![1, BN])
            .broadcast(shape![64, BN]);
        let v = sources
            .partition(shape![BN])
            .load([pid.0])
            .reshape(shape![1, BN])
            .broadcast(shape![64, BN]);
        let row: Tile<u64, { [64] }> = iota(shape![64]);
        let row = row.reshape(shape![64, 1]).broadcast(shape![64, BN]);
        let one: Tile<u64, { [64, BN] }> = constant(1u64, shape![64, BN]);
        let zero: Tile<u64, { [64, BN] }> = constant(0u64, shape![64, BN]);
        let wb = andi(shri(w, row), one);
        let lb = andi(shri(l, row), one);
        let db = andi(shri(d, row), one);
        let c8: Tile<i32, { [64, BN] }> = constant(8i32, shape![64, BN]);
        let c1: Tile<i32, { [64, BN] }> = constant(1i32, shape![64, BN]);
        let c0: Tile<i32, { [64, BN] }> = constant(0i32, shape![64, BN]);
        let value = select(cmpi(wb, zero, predicate::Equal), c8, c0)
            + select(cmpi(lb, zero, predicate::NotEqual), c1, c0);
        let value = select(cmpi(db, zero, predicate::NotEqual), value, c0);
        // Integer narrowing is not available in the pinned convert_tile, so route
        // through f32. The values 0, 8 and 9 are exact in f32, i8, f16 and FP8 E4M3.
        let value: Tile<f32, { [64, BN] }> = convert_tile(value);
        let rhs: Tile<E, { [64, BN] }> = convert_tile(value);
        let matrix = geometry.partition(shape![64, 64]).load([0, 0]);
        let initial: A = convert_scalar(0i32);
        let accumulator: Tile<A, { [64, BN] }> = broadcast_scalar(initial, shape![64, BN]);
        let sums = mma(matrix, rhs, accumulator);
        let sums: Tile<i32, { [64, BN] }> = convert_tile(sums);
        let z: Tile<i32, { [64, BN] }> = constant(0i32, shape![64, BN]);
        let c7: Tile<i32, { [64, BN] }> = constant(7i32, shape![64, BN]);
        let c72: Tile<i32, { [64, BN] }> = constant(72i32, shape![64, BN]);
        let bad = cmpi(sums, z, predicate::NotEqual);
        let win = ori(
            cmpi(andi(sums, c7), z, predicate::NotEqual),
            cmpi(sums, c72, predicate::Equal),
        );
        let source = cmpi(andi(shri(v, row), one), zero, predicate::NotEqual);
        let bit = one << row;
        let win_bits: Tile<u64, { [BN] }> = reduce_sum(select(andi(win, source), bit, zero), 0);
        let bad_bits: Tile<u64, { [BN] }> = reduce_sum(select(andi(bad, source), bit, zero), 0);
        let plane: Tile<u64, { [2] }> = iota(shape![2]);
        let plane = plane.reshape(shape![1, 2]).broadcast(shape![BN, 2]);
        let zero2: Tile<u64, { [BN, 2] }> = constant(0u64, shape![BN, 2]);
        let result = select(
            cmpi(plane, zero2, predicate::Equal),
            win_bits.reshape(shape![BN, 1]).broadcast(shape![BN, 2]),
            bad_bits.reshape(shape![BN, 1]).broadcast(shape![BN, 2]),
        );
        out.store(result);
    }
    // Knight reach is inlined in the entry: the pinned cuTile JIT does not
    // substitute const generics through a helper's return type.
    #[cutile::entry()]
    fn knight_bits<const B: i32>(
        out: &mut Tensor<u64, { [B, 2] }>,
        wins: &Tensor<u64, { [-1] }>,
        losses: &Tensor<u64, { [-1] }>,
        destinations: &Tensor<u64, { [-1] }>,
        sources: &Tensor<u64, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let w = wins.partition(shape![B]).load([pid.0]);
        let l = losses.partition(shape![B]).load([pid.0]);
        let d = destinations.partition(shape![B]).load([pid.0]);
        let v = sources.partition(shape![B]).load([pid.0]);
        let ma = constant(0x7f7f7f7f7f7f7f7fu64, shape![B]);
        // The pinned cuTile lowers u64 constants >= 2^63 incorrectly, so derive
        // high-bit masks from positive constants with exact shifts.
        let mb = ma << constant(1u64, shape![B]);
        let mc = constant(0x3f3f3f3f3f3f3f3fu64, shape![B]);
        let md = mc << constant(2u64, shape![B]);
        let all_ones = mb | constant(0x0101010101010101u64, shape![B]);
        let s6 = constant(6u64, shape![B]);
        let s10 = constant(10u64, shape![B]);
        let s15 = constant(15u64, shape![B]);
        let s17 = constant(17u64, shape![B]);
        let x = l & d;
        let win = (((x & ma) << s17)
            | ((x & mb) << s15)
            | ((x & mc) << s10)
            | ((x & md) << s6)
            | ((x & ma) >> s15)
            | ((x & mb) >> s17)
            | ((x & mc) >> s6)
            | ((x & md) >> s10))
            & v;
        let x = (w ^ all_ones) & d;
        let bad = (((x & ma) << s17)
            | ((x & mb) << s15)
            | ((x & mc) << s10)
            | ((x & md) << s6)
            | ((x & ma) >> s15)
            | ((x & mb) >> s17)
            | ((x & mc) >> s6)
            | ((x & md) >> s10))
            & v;
        let p: Tile<u64, { [2] }> = iota(shape![2]);
        let p = p.reshape(shape![1, 2]).broadcast(shape![B, 2]);
        let zero2: Tile<u64, { [B, 2] }> = constant(0u64, shape![B, 2]);
        out.store(select(
            cmpi(p, zero2, predicate::Equal),
            win.reshape(shape![B, 1]).broadcast(shape![B, 2]),
            bad.reshape(shape![B, 1]).broadcast(shape![B, 2]),
        ));
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q: usize = std::env::args()
        .nth(1)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(1 << 22);
    if q == 0 || !q.is_multiple_of(256) {
        return Err("fiber count must be a positive multiple of 256".into());
    }
    let mut seed = 9827u64;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut w: Vec<_> = (0..q).map(|_| random()).collect();
    let mut l: Vec<_> = (0..q).map(|i| random() & !w[i]).collect();
    let mut d: Vec<_> = (0..q).map(|_| random()).collect();
    let mut v: Vec<_> = (0..q).map(|_| random()).collect();
    let fixture_data = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../fixtures/six_fibers.json"
    ))?;
    let fixtures: Vec<serde_json::Value> = serde_json::from_slice(&fixture_data)?;
    let mut actual_fibers = 0;
    for fixture in fixtures.iter().filter(|f| f["piece"] == 2).take(q - 4) {
        let i = actual_fibers;
        w[i] = fixture["w"].as_str().ok_or("fixture win plane")?.parse()?;
        l[i] = fixture["l"].as_str().ok_or("fixture loss plane")?.parse()?;
        d[i] = fixture["destination_safe"]
            .as_str()
            .ok_or("fixture destination mask")?
            .parse()?;
        v[i] = fixture["source_valid"]
            .as_str()
            .ok_or("fixture source mask")?
            .parse()?;
        actual_fibers += 1;
    }
    for (offset, (win, loss, destination, source)) in [
        (0, u64::MAX, u64::MAX, u64::MAX),
        (u64::MAX, 0, u64::MAX, u64::MAX),
        (0, 0, 0, u64::MAX),
        (0, u64::MAX, u64::MAX, 0),
    ]
    .into_iter()
    .enumerate()
    {
        let index = actual_fibers + offset;
        w[index] = win;
        l[index] = loss;
        d[index] = destination;
        v[index] = source;
    }
    let device = cutile::cuda_core::Device::new(0)?;
    let stream = device.new_stream()?;
    let event_start = device.new_event()?;
    let event_end = device.new_event()?;
    let expected: Vec<_> = (0..q)
        .map(|i| stencil(w[i], l[i], d[i], v[i], 0, 2).unwrap())
        .collect();
    let mut flat = vec![0; 2 * q];
    for (i, p) in expected.iter().enumerate() {
        // Output layout [fiber, plane]: grid X carries the fiber blocks, because
        // grid Y is limited to 65,535 blocks (q / 32 exceeds it at 4,194,304).
        flat[2 * i] = p.winning;
        flat[2 * i + 1] = p.nonwinning;
    }
    let matrix: Vec<i8> = (0..64)
        .flat_map(|source| (0..64).map(move |dest| ((knight_reach(1 << source) >> dest) & 1) as i8))
        .collect();
    let fmatrix: Vec<half::f16> = matrix
        .iter()
        .map(|&x| half::f16::from_f32(x as f32))
        .collect();
    let fp8matrix: Vec<cutile::cuda_core::f8e4m3fn> = matrix
        .iter()
        .map(|&x| cutile::cuda_core::f8e4m3fn(if x == 0 { 0 } else { 0x38 }))
        .collect();
    let fp8matrix = api::copy_host_vec_to_device(&Arc::new(fp8matrix))
        .reshape(&[64, 64])
        .sync_on(&stream)?;
    let matrix = api::copy_host_vec_to_device(&Arc::new(matrix))
        .reshape(&[64, 64])
        .sync_on(&stream)?;
    let fmatrix = api::copy_host_vec_to_device(&Arc::new(fmatrix))
        .reshape(&[64, 64])
        .sync_on(&stream)?;
    let (hw, hl, hd, hv) = (w.clone(), l.clone(), d.clone(), v.clone());
    let w = api::copy_host_vec_to_device(&Arc::new(w)).sync_on(&stream)?;
    let l = api::copy_host_vec_to_device(&Arc::new(l)).sync_on(&stream)?;
    let d = api::copy_host_vec_to_device(&Arc::new(d)).sync_on(&stream)?;
    let v = api::copy_host_vec_to_device(&Arc::new(v)).sync_on(&stream)?;
    // Sentinel fill: a kernel that writes nothing must not inherit a prior mode's output.
    let sentinel = 0x5a5a_5a5a_5a5a_5a5au64;
    let mut out = api::full::<u64>(sentinel, &[q, 2]).sync_on(&stream)?;
    // Launch one mode either synchronously or asynchronously (no host sync).
    macro_rules! launch {
        ($mode:expr, $($exec:tt)+) => {
            match $mode {
                "packed_bits" => {
                    kernel::knight_bits((&mut out).partition([256, 2]), &w, &l, &d, &v).$($exec)+;
                }
                "int8_int32" => {
                    kernel::knight_tc((&mut out).partition([32, 2]), &matrix, &w, &l, &d, &v)
                        .generics(vec!["i8".into(), "i32".into(), "32".into()])
                        .$($exec)+;
                }
                "fp16_fp32" => {
                    kernel::knight_tc((&mut out).partition([32, 2]), &fmatrix, &w, &l, &d, &v)
                        .generics(vec!["f16".into(), "f32".into(), "32".into()])
                        .$($exec)+;
                }
                _ => {
                    kernel::knight_tc((&mut out).partition([32, 2]), &fp8matrix, &w, &l, &d, &v)
                        .generics(vec!["f8e4m3fn".into(), "f32".into(), "32".into()])
                        .$($exec)+;
                }
            }
        };
    }
    fn stats(mut x: Vec<f64>) -> serde_json::Value {
        x.sort_by(f64::total_cmp);
        let n = x.len();
        let median = if n % 2 == 1 {
            x[n / 2]
        } else {
            (x[n / 2 - 1] + x[n / 2]) / 2.0
        };
        json!({"n":n,"median":median,"min":x[0],"max":x[n-1],"spread_max_minus_min":x[n-1]-x[0],"samples":x})
    }
    let host_repetitions = 21;
    let event_batches = 15;
    let launches_per_batch = 50;
    let mut results = Vec::new();
    for mode in ["packed_bits", "int8_int32", "fp16_fp32", "fp8_fp32"] {
        if mode != "packed_bits" {
            out = api::full::<u64>(sentinel, &[q, 2]).sync_on(&stream)?;
        }
        let start = Instant::now();
        launch!(mode, sync_on(&stream)?);
        let first_launch_seconds = start.elapsed().as_secs_f64();
        let got = out.dup().to_host_vec().sync_on(&stream)?;
        if got != flat {
            let bad: Vec<usize> = (0..2 * q).filter(|&i| got[i] != flat[i]).collect();
            eprintln!("{mode}: {} of {} output words differ", bad.len(), 2 * q);
            for &i in bad.iter().take(6) {
                let (plane, f) = (i % 2, i / 2);
                eprintln!(
                    "  plane={plane} fiber={f} expected={:#018x} actual={:#018x}",
                    flat[i], got[i]
                );
            }
            if let Some(&i) = bad.first() {
                let f = i / 2;
                eprintln!(
                    "  first fiber inputs w={:#018x} l={:#018x} d={:#018x} v={:#018x}",
                    hw[f], hl[f], hd[f], hv[f]
                );
            }
            return Err(format!("{mode} disagrees with exact CPU oracle").into());
        }
        // Host scope: one synchronous launch per sample, including launch and sync.
        let mut host = Vec::new();
        for _ in 0..host_repetitions {
            let start = Instant::now();
            launch!(mode, sync_on(&stream)?);
            host.push(start.elapsed().as_secs_f64() * 1e3);
        }
        // Device scope: CUDA events around back-to-back asynchronous launches.
        // No host synchronization occurs between the two events.
        let mut device = Vec::new();
        let mut device_raw = Vec::new();
        let mut host_batch = Vec::new();
        for _ in 0..event_batches {
            let start = Instant::now();
            event_start.record(&stream)?;
            for _ in 0..launches_per_batch {
                // SAFETY: every input/output tensor outlives event_end.synchronize() below.
                unsafe {
                    launch!(mode, async_on(&stream)?);
                }
            }
            event_end.record(&stream)?;
            event_end.synchronize()?;
            host_batch.push(start.elapsed().as_secs_f64() * 1e3 / launches_per_batch as f64);
            let ms = event_start.elapsed_time(&event_end)? as f64 / launches_per_batch as f64;
            device_raw.push(ms);
            // A non-positive event interval is physically impossible for real work;
            // record it as invalid instead of using it in the statistics.
            if ms > 0.0 {
                device.push(ms);
            }
        }
        // event_end.synchronize() above has completed all launches. Validate again after the timed launches.
        if out.dup().to_host_vec().sync_on(&stream)? != flat {
            return Err(format!("{mode} output changed during timing").into());
        }
        let invalid = event_batches - device.len();
        let device_median_ms = stats(device.clone())["median"].as_f64().unwrap_or(f64::NAN);
        results.push(json!({
            "kernel": mode,
            "validated_against_cpu_oracle": true,
            "jit_plus_first_launch_plus_sync_seconds": first_launch_seconds,
            "host_ms_per_sync_launch": stats(host),
            "device_event_ms_per_launch_back_to_back": stats(device),
            "device_event_raw_samples_ms": device_raw,
            "device_event_invalid_nonpositive_samples": invalid,
            "host_ms_per_launch_back_to_back_batch": stats(host_batch),
            "device_event_scope": "CUDA events around back-to-back async launches; includes on-device inter-kernel gaps, excludes host sync and transfers",
            "useful_source_rows_per_launch": q * 64,
            "source_rows_per_second_device_median": (q * 64) as f64 / (device_median_ms / 1e3),
            "nominal_bytes_per_launch": q * 48,
        }));
    }
    println!(
        "{}",
        json!({"results":results,"selected_device":std::env::var("CUDA_VISIBLE_DEVICES").unwrap_or_default(),"independent_fibers":q,"actual_six_piece_geometry_fixtures":actual_fibers,"special_fibers":["all_win","all_loss_eight_loss_carry","no_move","invalid_source"],"tensor_columns":32,"geometry_shared":true,"includes_packed_input_expansion":true,"includes_bitpacking_outputs":true,"includes_move_masks_generation":false,"includes_global_fixpoint":false,"whole_material_solved":false,"encoding":"0/8/9 with eight-loss carry","fp8_fast_acc_enabled":false,"static_legal_mask_cached_separately":true,"nominal_resident_bytes_per_source_row":0.75})
    );
    Ok(())
}
