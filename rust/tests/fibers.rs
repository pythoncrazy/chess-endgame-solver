use serde::Deserialize;
use tablebase_core::fiber::{
    Predicates, encoded_knight, encoded_knight8, encoded_knight16, knight_reach, stencil,
};
#[derive(Deserialize)]
struct Expected {
    winning: String,
    nonwinning: String,
    legal: String,
}
#[derive(Deserialize)]
struct Fixture {
    piece: u8,
    w: String,
    l: String,
    destination_safe: String,
    source_valid: String,
    occupied: String,
    expected: Expected,
}
#[test]
fn independent_six_piece_legal_move_fibers() {
    let fixtures: Vec<Fixture> = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/six_fibers.json"
        ))
        .unwrap(),
    )
    .unwrap();
    for f in fixtures {
        let (w, l, d, v, o) = (
            f.w.parse().unwrap(),
            f.l.parse().unwrap(),
            f.destination_safe.parse().unwrap(),
            f.source_valid.parse().unwrap(),
            f.occupied.parse().unwrap(),
        );
        let expected = Predicates {
            winning: f.expected.winning.parse().unwrap(),
            nonwinning: f.expected.nonwinning.parse().unwrap(),
            legal: f.expected.legal.parse().unwrap(),
        };
        assert_eq!(stencil(w, l, d, v, o, f.piece).unwrap(), expected);
        if f.piece == 2 {
            assert_eq!(encoded_knight(w, l, d, v).unwrap(), expected);
            assert_eq!(encoded_knight16(w, l, d, v).unwrap(), expected);
            assert_eq!(encoded_knight8(w, l, d, v).unwrap(), expected);
        }
    }
}
#[test]
fn fp8_carry_and_all_child_assignments() {
    let source = 27;
    let children: Vec<_> = (0..64)
        .filter(|&s| knight_reach(1 << source) >> s & 1 != 0)
        .collect();
    assert_eq!(children.len(), 8);
    for mut code in 0..65536u32 {
        let (mut w, mut l, mut d) = (0, 0, 0);
        for &child in &children {
            let state = code & 3;
            code >>= 2;
            if state != 0 {
                d |= 1 << child;
            }
            if state == 1 {
                w |= 1 << child;
            }
            if state == 3 {
                l |= 1 << child;
            }
        }
        let expected = stencil(w, l, d, 1 << source, 0, 2).unwrap();
        assert_eq!(encoded_knight8(w, l, d, 1 << source).unwrap(), expected);
        assert_eq!(encoded_knight16(w, l, d, 1 << source).unwrap(), expected);
    }
    assert!(encoded_knight8(1, 1, 0, 0).is_err());
}
#[test]
fn radix_edges_and_sparse_geometry() {
    for s in 0..64 {
        let row = knight_reach(1 << s);
        assert!(row.count_ones() <= 8);
        for group in 0..16 {
            assert!(((row >> (group * 4)) & 15).count_ones() <= 2);
        }
    }
    for (w, l) in [(u64::MAX, 0), (0, u64::MAX), (0, 0)] {
        assert_eq!(
            encoded_knight(w, l, u64::MAX, u64::MAX).unwrap(),
            stencil(w, l, u64::MAX, u64::MAX, 0, 2).unwrap()
        );
    }
    assert!(encoded_knight(1, 1, 0, 0).is_err());
    assert!(stencil(0, 0, 1, 0, 1, 2).is_err());
}
