//! Convert pawnless tables (v1/v2) to the compressed v3 format; `--check`
//! decodes each output and requires it to equal the input exactly.
use std::{fs::File, io::BufWriter, path::PathBuf, time::Instant};
use tablebase_core::packed::{self, Canonical, PackedTable};

fn main() -> Result<(), String> {
    let usage = "usage: tb_pack [--level L] [--threads T] [--check] OUTPUT_DIRECTORY TABLE...";
    let mut args = std::env::args().skip(1).peekable();
    let (mut level, mut threads, mut check) = (19, 24, false);
    while let Some(flag) = args.next_if(|a| a.starts_with("--")) {
        match flag.as_str() {
            "--level" => {
                level = args
                    .next()
                    .ok_or(usage)?
                    .parse()
                    .map_err(|e| format!("{e}"))?
            }
            "--threads" => {
                threads = args
                    .next()
                    .ok_or(usage)?
                    .parse()
                    .map_err(|e| format!("{e}"))?
            }
            "--check" => check = true,
            _ => return Err(usage.into()),
        }
    }
    let output = PathBuf::from(args.next().ok_or(usage)?);
    std::fs::create_dir_all(&output).map_err(|e| e.to_string())?;
    let mut report = Vec::new();
    for input in args {
        let t0 = Instant::now();
        let (table, source_digest) = packed::load_table(input.as_ref(), threads)?;
        let read_seconds = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let stored = Canonical::from_table(&table);
        let path = output.join(format!("{}.tbz", table.material.name()));
        let file = File::create(&path).map_err(|e| e.to_string())?;
        let header = packed::write(
            &stored,
            BufWriter::new(file),
            concat!("tb_pack ", env!("CARGO_PKG_VERSION")),
            level,
            threads,
        )?;
        let pack_seconds = t1.elapsed().as_secs_f64();
        let mut row = serde_json::json!({
            "material": header.material, "source": input, "source_sha256": source_digest,
            "file": path, "bytes": std::fs::metadata(&path).map_err(|e| e.to_string())?.len(),
            "sections": header.sections, "content_sha256": header.content_sha256,
            "read_seconds": read_seconds, "pack_seconds": pack_seconds,
        });
        if check {
            let t2 = Instant::now();
            let decoded = PackedTable::open(&path, true)?.decode(threads)?;
            let equal = decoded == table;
            row["roundtrip_equal"] = equal.into();
            row["decode_seconds"] = t2.elapsed().as_secs_f64().into();
            if !equal {
                println!("{row}");
                return Err(format!("{}: decoded table differs", header.material));
            }
        }
        eprintln!("{row}");
        report.push(row);
    }
    println!(
        "{}",
        serde_json::json!({"tool": "tb_pack", "level": level, "tables": report})
    );
    Ok(())
}
