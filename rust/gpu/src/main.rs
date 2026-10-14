//! Resident exact bitmask propagation microbenchmark; not a chess generator.
use cutile::prelude::*;
use std::{sync::Arc, time::Instant};

#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry()]
    fn propagate<const B: i32>(
        out: &mut Tensor<i32, { [B] }>,
        state: &Tensor<i32, { [-1] }>,
        losing: &Tensor<u64, { [-1] }>,
        winning: &Tensor<u64, { [-1] }>,
        legal: &Tensor<u64, { [-1] }>,
    ) {
        let s = state.load_like(out);
        let l = losing.load_like(out);
        let w = winning.load_like(out);
        let m = legal.load_like(out);
        let zero = constant(0u64, out.shape());
        let any_loss = cmpi(andi(l, m), zero, predicate::NotEqual);
        let all_win = andi(
            cmpi(andi(w, m), m, predicate::Equal),
            cmpi(m, zero, predicate::NotEqual),
        );
        let unknown = cmpi(s, constant(0i32, out.shape()), predicate::Equal);
        let updated = select(
            any_loss,
            constant(1i32, out.shape()),
            select(all_win, constant(-1i32, out.shape()), s),
        );
        out.store(select(unknown, updated, s));
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 1 << 20;
    let mut seed = 93711u64;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let legal: Vec<_> = (0..n)
        .map(|i| if i % 17 == 0 { 0 } else { random() })
        .collect();
    let winning: Vec<_> = (0..n)
        .map(|i| if i % 7 == 0 { legal[i] } else { random() })
        .collect();
    let losing: Vec<_> = (0..n)
        .map(|i| {
            if i % 3 == 0 {
                0
            } else {
                random() & !winning[i]
            }
        })
        .collect();
    let state: Vec<i32> = (0..n).map(|i| if i % 13 == 0 { 1 } else { 0 }).collect();
    let expected: Vec<_> = (0..n)
        .map(|i| {
            match tablebase_core::retrograde::update_row(
                state[i] == 0,
                losing[i],
                winning[i],
                legal[i],
            ) {
                Some(tablebase_core::retrograde::Wdl::Win) => 1,
                Some(tablebase_core::retrograde::Wdl::Loss) => -1,
                _ => state[i],
            }
        })
        .collect();
    let state = api::copy_host_vec_to_device(&Arc::new(state)).sync()?;
    let losing = api::copy_host_vec_to_device(&Arc::new(losing)).sync()?;
    let winning = api::copy_host_vec_to_device(&Arc::new(winning)).sync()?;
    let legal = api::copy_host_vec_to_device(&Arc::new(legal)).sync()?;
    let mut output = api::zeros::<i32>(&[n]).sync()?;
    // First launch includes JIT; validation precedes timing. Reuse all allocations.
    kernels::propagate(
        (&mut output).partition([256]),
        &state,
        &losing,
        &winning,
        &legal,
    )
    .sync()?;
    let actual = output.dup().to_host_vec().sync()?;
    if actual != expected {
        return Err("GPU/CPU exact propagation mismatch".into());
    }
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        kernels::propagate(
            (&mut output).partition([256]),
            &state,
            &losing,
            &winning,
            &legal,
        )
        .sync()?;
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "rows={n} iterations={iterations} seconds={elapsed:.6} rows_per_second={:.3e} nominal_gb_s={:.2} timing=host_wall_including_launch_and_sync",
        n as f64 * iterations as f64 / elapsed,
        n as f64 * iterations as f64 * 32.0 / elapsed / 1e9
    );
    Ok(())
}
