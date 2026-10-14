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
        out: &mut Tensor<u64, { [2, BN] }>,
        geometry: &Tensor<E, { [64, 64] }>,
        wins: &Tensor<u64, { [-1] }>,
        losses: &Tensor<u64, { [-1] }>,
        destinations: &Tensor<u64, { [-1] }>,
        sources: &Tensor<u64, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let w = wins
            .partition(shape![BN])
            .load([pid.1])
            .reshape(shape![1, BN])
            .broadcast(shape![64, BN]);
        let l = losses
            .partition(shape![BN])
            .load([pid.1])
            .reshape(shape![1, BN])
            .broadcast(shape![64, BN]);
        let d = destinations
            .partition(shape![BN])
            .load([pid.1])
            .reshape(shape![1, BN])
            .broadcast(shape![64, BN]);
        let v = sources
            .partition(shape![BN])
            .load([pid.1])
            .reshape(shape![1, BN])
            .broadcast(shape![64, BN]);
        let row: Tile<u64, { [64] }> = iota(shape![64]);
        let row = row.reshape(shape![64, 1]).broadcast(shape![64, BN]);
        let one = constant(1u64, shape![64, BN]);
        let zero = constant(0u64, shape![64, BN]);
        let wb = andi(shri(w, row), one);
        let lb = andi(shri(l, row), one);
        let db = andi(shri(d, row), one);
        let value = select(
            cmpi(wb, zero, predicate::Equal),
            constant(8i32, shape![64, BN]),
            constant(0i32, shape![64, BN]),
        ) + select(
            cmpi(lb, zero, predicate::NotEqual),
            constant(1i32, shape![64, BN]),
            constant(0i32, shape![64, BN]),
        );
        let value = select(
            cmpi(db, zero, predicate::NotEqual),
            value,
            constant(0i32, shape![64, BN]),
        );
        let rhs: Tile<E, { [64, BN] }> = convert_tile(value);
        let matrix = geometry.partition(shape![64, 64]).load([0, 0]);
        let initial: A = convert_scalar(0i32);
        let accumulator: Tile<A, { [64, BN] }> = broadcast_scalar(initial, shape![64, BN]);
        let sums = mma(matrix, rhs, accumulator);
        let sums: Tile<i32, { [64, BN] }> = convert_tile(sums);
        let z = constant(0i32, shape![64, BN]);
        let bad = cmpi(sums, z, predicate::NotEqual);
        let win = ori(
            cmpi(
                andi(sums, constant(7i32, shape![64, BN])),
                z,
                predicate::NotEqual,
            ),
            cmpi(sums, constant(72i32, shape![64, BN]), predicate::Equal),
        );
        let source = cmpi(andi(shri(v, row), one), zero, predicate::NotEqual);
        let bit = shli(one, row, overflow::None);
        let win_bits: Tile<u64, { [BN] }> = reduce_sum(select(andi(win, source), bit, zero), 0);
        let bad_bits: Tile<u64, { [BN] }> = reduce_sum(select(andi(bad, source), bit, zero), 0);
        let plane: Tile<u64, { [2] }> = iota(shape![2]);
        let plane = plane.reshape(shape![2, 1]).broadcast(shape![2, BN]);
        let result = select(
            cmpi(plane, constant(0u64, shape![2, BN]), predicate::Equal),
            win_bits.reshape(shape![1, BN]).broadcast(shape![2, BN]),
            bad_bits.reshape(shape![1, BN]).broadcast(shape![2, BN]),
        );
        out.store(result);
    }
    fn reach<const B: i32>(x: Tile<u64, { [B] }>) -> Tile<u64, { [B] }> {
        let a = constant(0x7f7f7f7f7f7f7f7fu64, shape![B]);
        let b = constant(0xfefefefefefefefeu64, shape![B]);
        let c = constant(0x3f3f3f3f3f3f3f3fu64, shape![B]);
        let d = constant(0xfcfcfcfcfcfcfcfcu64, shape![B]);
        ((x & a) << constant(17u64, shape![B]))
            | ((x & b) << constant(15u64, shape![B]))
            | ((x & c) << constant(10u64, shape![B]))
            | ((x & d) << constant(6u64, shape![B]))
            | ((x & a) >> constant(15u64, shape![B]))
            | ((x & b) >> constant(17u64, shape![B]))
            | ((x & c) >> constant(6u64, shape![B]))
            | ((x & d) >> constant(10u64, shape![B]))
    }
    #[cutile::entry()]
    fn knight_bits<const B: i32>(
        out: &mut Tensor<u64, { [2, B] }>,
        wins: &Tensor<u64, { [-1] }>,
        losses: &Tensor<u64, { [-1] }>,
        destinations: &Tensor<u64, { [-1] }>,
        sources: &Tensor<u64, { [-1] }>,
    ) {
        let pid = get_tile_block_id();
        let w = wins.partition(shape![B]).load([pid.1]);
        let l = losses.partition(shape![B]).load([pid.1]);
        let d = destinations.partition(shape![B]).load([pid.1]);
        let v = sources.partition(shape![B]).load([pid.1]);
        let win = reach(l & d) & v;
        let bad = reach((w ^ constant(u64::MAX, shape![B])) & d) & v;
        let p: Tile<u64, { [2] }> = iota(shape![2]);
        let p = p.reshape(shape![2, 1]).broadcast(shape![2, B]);
        out.store(select(
            cmpi(p, constant(0u64, shape![2, B]), predicate::Equal),
            win.reshape(shape![1, B]).broadcast(shape![2, B]),
            bad.reshape(shape![1, B]).broadcast(shape![2, B]),
        ));
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let q: usize = std::env::args()
        .nth(1)
        .map(|x| x.parse())
        .transpose()?
        .unwrap_or(1 << 22);
    if q == 0 || q % 256 != 0 {
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
        flat[i] = p.winning;
        flat[q + i] = p.nonwinning;
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
    let w = api::copy_host_vec_to_device(&Arc::new(w)).sync_on(&stream)?;
    let l = api::copy_host_vec_to_device(&Arc::new(l)).sync_on(&stream)?;
    let d = api::copy_host_vec_to_device(&Arc::new(d)).sync_on(&stream)?;
    let v = api::copy_host_vec_to_device(&Arc::new(v)).sync_on(&stream)?;
    let mut out = api::zeros::<u64>(&[2, q]).sync_on(&stream)?;
    let iterations = 20;
    let mut results = Vec::new();
    for mode in ["packed_bits", "int8_int32", "fp16_fp32", "fp8_fp32"] {
        for iteration in 0..=iterations {
            let start = Instant::now();
            event_start.record(&stream)?;
            match mode {
                "packed_bits" => {
                    kernel::knight_bits((&mut out).partition([2, 256]), &w, &l, &d, &v)
                        .sync_on(&stream)?;
                }
                "int8_int32" => {
                    kernel::knight_tc((&mut out).partition([2, 32]), &matrix, &w, &l, &d, &v)
                        .generics(vec!["i8".into(), "i32".into(), "32".into()])
                        .sync_on(&stream)?;
                }
                "fp16_fp32" => {
                    kernel::knight_tc((&mut out).partition([2, 32]), &fmatrix, &w, &l, &d, &v)
                        .generics(vec!["f16".into(), "f32".into(), "32".into()])
                        .sync_on(&stream)?;
                }
                _ => {
                    kernel::knight_tc((&mut out).partition([2, 32]), &fp8matrix, &w, &l, &d, &v)
                        .generics(vec!["f8e4m3fn".into(), "f32".into(), "32".into()])
                        .sync_on(&stream)?;
                }
            }
            event_end.record(&stream)?;
            event_end.synchronize()?;
            let device_ms = event_start.elapsed_time(&event_end)?;
            if iteration == 0 {
                if out.dup().to_host_vec().sync_on(&stream)? != flat {
                    return Err(format!("{mode} disagrees with exact CPU oracle").into());
                }
            } else {
                results.push(json!({"kernel":mode,"iteration":iteration,"seconds":start.elapsed().as_secs_f64(),"device_event_ms_including_launch_gaps":device_ms,"useful_source_rows":q*64}));
            }
        }
    }
    println!(
        "{}",
        json!({"results":results,"independent_fibers":q,"actual_six_piece_geometry_fixtures":actual_fibers,"tensor_columns":32,"geometry_shared":true,"includes_packed_input_expansion":true,"includes_bitpacking_outputs":true,"includes_move_masks_generation":false,"includes_global_fixpoint":false,"includes_launch_sync":true,"whole_material_solved":false,"encoding":"0/8/9 with eight-loss carry","fp8_fast_acc_enabled":false,"static_legal_mask_cached_separately":true,"nominal_resident_bytes_per_source_row":0.75})
    );
    Ok(())
}
