use tablebase_core::packed::{
    self, Canonical, Codec, Layout, PackedTable, RankSides, WriteOptions,
};
use tablebase_core::pawnless::{Material, Solved, Symmetry, Table, solve, solve_tree};

fn solved(name: &str) -> Table {
    let mut solved = Solved::new();
    let mut cpu = |m: &Material, lower: &[Option<&Table>]| solve(m, lower, 8);
    (*solve_tree(&Material::parse(name).unwrap(), &mut solved, &mut cpu).unwrap()).clone()
}

fn pack(table: &Table, path: &std::path::Path, options: WriteOptions) -> PackedTable {
    let file = std::fs::File::create(path).unwrap();
    let header = packed::write(&Canonical::from_table(table), file, "test", options, 4).unwrap();
    let opened = PackedTable::open(path, true).unwrap();
    assert_eq!(opened.header, header);
    opened
}

/// There is exactly one stored position for each symmetry class. The eight images
/// of any position share one stored index, with identical pieces in any order.
#[test]
fn stored_index_is_symmetric_and_onto() {
    for name in ["KQvK", "KNNvK", "KRvKB"] {
        let material = Material::parse(name).unwrap();
        let layout = Layout::new(&material);
        let m = material.skeleton_len();
        let mut stored = std::collections::HashSet::new();
        for word in (0..material.words()).step_by(7) {
            let (black, squares) = material.decode(word);
            for bk in (0..64u8).step_by(5) {
                let p = layout.position(black, &squares[..m], bk);
                assert!(p < layout.words() * 64);
                stored.insert(p);
                for g in Symmetry::all() {
                    let mut image: Vec<u8> = squares[..m].iter().map(|&s| g.square(s)).collect();
                    if name == "KNNvK" {
                        image.swap(1, 2);
                    }
                    assert_eq!(layout.position(black, &image, g.square(bk)), p, "{name}");
                }
            }
        }
        assert!(!stored.is_empty());
    }
}

#[test]
fn packed_roundtrip_probe_and_identical_pieces() {
    let dir = std::env::temp_dir().join(format!("packed-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let both = WriteOptions {
        level: 3,
        wdl_codec: Codec::Zstd,
        rank_codec: Codec::Zstd,
        rank_sides: RankSides::Both,
        ..Default::default()
    };
    let white = WriteOptions {
        level: 3,
        ..Default::default()
    };
    let cm = WriteOptions {
        wdl_codec: Codec::Cm,
        rank_codec: Codec::Cm,
        ..Default::default()
    };
    let cm_small = WriteOptions {
        cm_wdl_words: 7,
        cm_rank_words: 3,
        ..cm
    };
    for (name, options) in [
        ("KQvK", both),
        ("KQvK", white),
        ("KvKR", white),
        ("KNNvK", white),
        ("KRvKB", white),
        ("KQvK", cm),
        ("KvKR", cm),
        ("KNNvK", cm),
        ("KRvKB", cm),
        ("KRvKB", cm_small),
    ] {
        let table = solved(name);
        let path = dir.join(format!("{name}.tbz"));
        let packed = pack(&table, &path, options);
        assert_eq!(
            packed.decode(4).unwrap(),
            table,
            "{name} {options:?} decode"
        );
        let m = table.material.skeleton_len();
        let step = [1, 1, 1, 97, 9973][m];
        for word in (0..table.valid.len()).step_by(step) {
            let (black, squares) = table.material.decode(word);
            for bk in 0..64u8 {
                let value = packed.probe(black, &squares[..m], bk).unwrap();
                match table.wdl(word, bk) {
                    None => assert_eq!(value, None),
                    Some(wdl) => {
                        let value = value.unwrap();
                        assert_eq!(value.wdl, wdl);
                        let rank = table.rank[word * 64 + usize::from(bk)];
                        assert_eq!(value.rank, (wdl != 0).then_some(rank));
                    }
                }
            }
        }
    }
    // Authentication rejects corruption.
    let path = dir.join("KQvK.tbz");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&path, &bytes).unwrap();
    assert!(PackedTable::open(&path, true).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Rewrite one header field of a packed file.
fn with_header(bytes: &[u8], edit: impl Fn(&mut serde_json::Value)) -> Vec<u8> {
    let len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let mut header: serde_json::Value = serde_json::from_slice(&bytes[16..16 + len]).unwrap();
    edit(&mut header);
    let json = serde_json::to_vec(&header).unwrap();
    let mut out = bytes[..8].to_vec();
    out.extend((json.len() as u64).to_le_bytes());
    out.extend(json);
    out.extend(&bytes[16 + len..]);
    out
}

/// A header with sizes that disagree with the material or the file is an error.
/// It must not cause a panic or a large allocation.
#[test]
fn bad_headers_are_errors() {
    let dir = std::env::temp_dir().join(format!("packed-bad-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let table = solved("KQvK");
    let zstd = WriteOptions {
        level: 3,
        ..Default::default()
    };
    let cm = WriteOptions {
        wdl_codec: Codec::Cm,
        rank_codec: Codec::Cm,
        ..zstd
    };
    let huge = u64::MAX / 3;
    let cases: [(&str, &str, u64); 9] = [
        ("", "rank_width", 3),
        ("0", "blocks", 1 << 40),
        ("0", "raw_bytes", 16),
        ("1", "raw_bytes", 1 << 40),
        ("1", "offset", huge),
        ("0", "model_bytes", huge),
        ("1", "index_bytes", huge),
        ("0", "block_words", 0),
        ("1", "blocks", u64::from(u32::MAX)),
    ];
    for (codec, options) in [("zstd", zstd), ("cm", cm)] {
        let path = dir.join(format!("{codec}.tbz"));
        pack(&table, &path, options);
        let bytes = std::fs::read(&path).unwrap();
        let bad = dir.join("bad.tbz");
        for (section, field, value) in cases {
            let present = std::cell::Cell::new(false);
            let edited = with_header(&bytes, |h| {
                let target = match section {
                    "" => &mut *h,
                    s => &mut h["sections"][s.parse::<usize>().unwrap()],
                };
                if target.get(field).is_some() {
                    target[field] = value.into();
                    present.set(true);
                }
            });
            if !present.get() {
                continue;
            }
            std::fs::write(&bad, &edited).unwrap();
            assert!(
                PackedTable::open(&bad, false).is_err(),
                "{codec} {section} {field} {value}"
            );
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
