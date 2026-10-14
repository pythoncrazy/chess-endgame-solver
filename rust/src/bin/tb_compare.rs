//! Compare the valid/win/loss planes and proof ranks of two pawnless table
//! files of any format version.
use tablebase_core::{packed::load_table, pawnless::Table};

fn load(path: &str) -> Result<Table, String> {
    Ok(load_table(path.as_ref(), 24)?.0)
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: tb_compare TABLE_A TABLE_B";
    let (a, b) = (
        load(&args.next().ok_or(usage)?)?,
        load(&args.next().ok_or(usage)?)?,
    );
    if a.material != b.material {
        return Err("tables hold different materials".into());
    }
    let mut differing_words = 0u64;
    let mut first = None;
    for w in 0..a.valid.len() {
        let diff = (a.valid[w] ^ b.valid[w]) | (a.win[w] ^ b.win[w]) | (a.loss[w] ^ b.loss[w]);
        if diff != 0 {
            differing_words += 1;
            first.get_or_insert((w, diff.trailing_zeros()));
        }
    }
    let ranks_compared = !a.rank.is_empty() && !b.rank.is_empty();
    let rank_differences = if ranks_compared {
        a.rank.iter().zip(&b.rank).filter(|(x, y)| x != y).count()
    } else {
        0
    };
    let equal = differing_words == 0 && rank_differences == 0;
    println!(
        "{}",
        serde_json::json!({
            "material": a.material.name(), "words": a.valid.len(),
            "wdl_planes_equal": differing_words == 0, "differing_words": differing_words,
            "first_difference_word_and_black_king": first,
            "ranks_compared": ranks_compared, "rank_differences": rank_differences,
            "counts_a": a.counts(), "counts_b": b.counts(),
        })
    );
    if equal {
        Ok(())
    } else {
        Err("tables differ".into())
    }
}
