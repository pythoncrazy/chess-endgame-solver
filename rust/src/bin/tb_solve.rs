//! CPU reference solve of a pawnless material and all its capture dependencies.
use std::{fs::File, io::BufWriter, path::PathBuf, time::Instant};
use tablebase_core::pawnless::{Material, Solved, solve, solve_tree};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: tb_solve MATERIAL OUTPUT_DIRECTORY [THREADS]";
    let material = Material::parse(&args.next().ok_or(usage)?)?;
    let output = PathBuf::from(args.next().ok_or(usage)?);
    let threads = match args.next() {
        Some(t) => t.parse().map_err(|e| format!("threads: {e}"))?,
        None => 8,
    };
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let mut records = Vec::new();
    let mut solve_one = |m: &Material, lower: &[Option<&_>]| {
        let start = Instant::now();
        let table = solve(m, lower, threads)?;
        let seconds = start.elapsed().as_secs_f64();
        let path = output.join(format!("{}.cpu.tbp", m.name()));
        let file = File::create(&path).map_err(|e| e.to_string())?;
        let header = table.write(
            BufWriter::new(file),
            concat!("tb_solve ", env!("CARGO_PKG_VERSION")),
        )?;
        records.push(serde_json::json!({
            "material": m.name(), "seconds": seconds, "rounds": table.rounds,
            "counts_white_to_move": table.counts()[0], "counts_black_to_move": table.counts()[1],
            "file": path, "payload_sha256": header.payload_sha256,
        }));
        Ok(table)
    };
    solve_tree(&material, &mut Solved::new(), &mut solve_one)?;
    println!(
        "{}",
        serde_json::json!({"solver": "cpu-reference", "threads": threads, "tables": records})
    );
    Ok(())
}
