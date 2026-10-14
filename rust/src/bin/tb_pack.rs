//! Convert pawnless tables (v1/v2/v3/v4) to the compressed v4 format;
//! `--check` decodes each output and requires it to equal the input exactly.
use std::{fs::File, io::BufWriter, path::PathBuf, time::Instant};
use tablebase_core::packed::{self, Canonical, Codec, PackedTable, RankSides, WriteOptions};

fn main() -> Result<(), String> {
    let usage = "usage: tb_pack [--level L] [--wdl-codec zstd|lzma] [--rank-codec zstd|lzma] \
                 [--ranks both|white] [--threads T] [--check] [--probes N] OUTPUT_DIRECTORY TABLE...";
    let mut args = std::env::args().skip(1).peekable();
    let (mut options, mut threads, mut check, mut probes) =
        (WriteOptions::default(), 24, false, 0usize);
    let codec = |v: Option<String>| match v.as_deref() {
        Some("zstd") => Ok(Codec::Zstd),
        Some("lzma") => Ok(Codec::Lzma),
        _ => Err(usage.to_string()),
    };
    while let Some(flag) = args.next_if(|a| a.starts_with("--")) {
        match flag.as_str() {
            "--wdl-codec" => options.wdl_codec = codec(args.next())?,
            "--rank-codec" => options.rank_codec = codec(args.next())?,
            "--ranks" => {
                options.rank_sides = match args.next().as_deref() {
                    Some("both") => RankSides::Both,
                    Some("white") => RankSides::White,
                    _ => return Err(usage.into()),
                }
            }
            "--level" => {
                options.level = args
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
            "--probes" => {
                probes = args
                    .next()
                    .ok_or(usage)?
                    .parse()
                    .map_err(|e| format!("{e}"))?
            }
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
            options,
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
            if probes > 0 {
                let m = table.material.skeleton_len();
                let mut seed = 0x9e37_79b9_7f4a_7c15u64;
                let mut sample = Vec::with_capacity(probes);
                while sample.len() < probes {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let (word, bk) = (
                        (seed >> 20) as usize % table.valid.len(),
                        (seed >> 8) as u8 & 63,
                    );
                    if table.valid[word] >> bk & 1 != 0 {
                        sample.push((word, bk));
                    }
                }
                let mut times = Vec::with_capacity(probes);
                for pass in 0..2 {
                    let opened = PackedTable::open(&path, false)?;
                    for &(word, bk) in &sample {
                        let (black, squares) = table.material.decode(word);
                        let t = Instant::now();
                        let value = opened
                            .probe(black, &squares[..m], bk)?
                            .ok_or("legal position probed illegal")?;
                        if pass == 0 {
                            times.push(t.elapsed().as_secs_f64() * 1e6);
                        }
                        let wdl = table.wdl(word, bk).expect("legal");
                        let rank = table.rank.get(word * 64 + usize::from(bk));
                        if value.wdl != wdl || value.rank != rank.copied().filter(|_| wdl != 0) {
                            return Err(format!(
                                "{}: probe disagrees at word {word}",
                                header.material
                            ));
                        }
                    }
                }
                times.sort_by(f64::total_cmp);
                let q = |f: f64| times[((times.len() - 1) as f64 * f) as usize];
                row["probe_us"] = serde_json::json!({
                    "n": probes, "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": q(1.0),
                    "mean": times.iter().sum::<f64>() / probes as f64,
                });
            }
        }
        eprintln!("{row}");
        report.push(row);
    }
    println!(
        "{}",
        serde_json::json!({"tool": "tb_pack", "options": format!("{options:?}"), "tables": report})
    );
    Ok(())
}
