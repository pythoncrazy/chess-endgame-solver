use tablebase_core::fiber::knight_reach;
use tablebase_core::pawnless::{
    Geometry, Material, Role, Solved, Table, king_reach, solve, solve_tree,
};

#[test]
fn material_names_indices_and_captures() {
    let m = Material::parse("KQBNvKR").unwrap();
    assert_eq!(m.name(), "KQBNvKR");
    assert_eq!(m.skeleton_len(), 5);
    assert_eq!(m.words(), 2 * 64usize.pow(5));
    assert_eq!(m.without(4).unwrap().name(), "KQBNvK");
    assert_eq!(m.without(1).unwrap().name(), "KBNvKR");
    assert!(m.without(0).is_err());
    for bad in ["KQK", "QvK", "KPvK", "KQvKK", "KQRBNvKQ"] {
        assert!(Material::parse(bad).is_err(), "{bad}");
    }
    for word in [0, 1, 12_345, m.words() / 2, m.words() - 1] {
        let (black, squares) = m.decode(word);
        assert_eq!(m.encode(black, &squares[..5]), word);
    }
}

#[test]
fn geometry_matches_bitboard_shifts() {
    let g = Geometry::new();
    for s in 0..64u8 {
        assert_eq!(g.attacks(Role::Knight, s, 0), knight_reach(1 << s));
        assert_eq!(g.king(s), king_reach(1 << s));
        let rook = g.attacks(Role::Rook, s, 0);
        assert_eq!(rook.count_ones(), 14);
        let queen = g.attacks(Role::Queen, s, 0);
        assert_eq!(queen, rook | g.attacks(Role::Bishop, s, 0));
    }
    // a1 rook blocked on a4 and d1; between excludes both endpoints.
    let occupied = (1 << 24) | (1 << 3);
    assert_eq!(
        g.attacks(Role::Rook, 0, occupied),
        (1 << 8) | (1 << 16) | (1 << 24) | 0b1110
    );
    assert_eq!(g.between(0, 63), 0x0040_2010_0804_0200);
    assert_eq!(g.between(0, 17), 0);
}

fn solved_kqvk() -> Table {
    let mut solved = Solved::new();
    let mut cpu = |m: &Material, lower: &[Option<&Table>]| solve(m, lower, 4);
    let table = solve_tree(&Material::parse("KQvK").unwrap(), &mut solved, &mut cpu).unwrap();
    assert!(solved.contains_key("KvK"));
    (*table).clone()
}

#[test]
fn kqvk_counts_and_known_positions() {
    // Counts verified against Syzygy by verify_pawnless (all legal positions).
    let t = solved_kqvk();
    let [white, black] = t.counts();
    assert_eq!((white.legal, white.win, white.draw), (144_508, 144_508, 0));
    assert_eq!(
        (black.legal, black.loss, black.draw),
        (223_944, 200_896, 23_048)
    );
    assert_eq!(t.rounds, 20);
    let m = &t.material;
    // 7k/6Q1/6K1/8/8/8/8/8 b: checkmate, rank 0.
    let mate = m.encode(true, &[46, 54]);
    assert_eq!(t.wdl(mate, 63), Some(-2));
    assert_eq!(t.rank[mate * 64 + 63], 0);
    // 7k/5Q2/6K1/8/8/8/8/8 b: stalemate.
    assert_eq!(t.wdl(m.encode(true, &[46, 53]), 63), Some(0));
    // Black king adjacent to the white king is illegal.
    assert_eq!(t.wdl(m.encode(true, &[0, 30]), 1), None);
    // Black to move can capture an unprotected adjacent queen: draw.
    assert_eq!(t.wdl(m.encode(true, &[0, 36]), 35), Some(0));
}

#[test]
fn table_file_roundtrip_and_rejections() {
    let t = solved_kqvk();
    let mut bytes = Vec::new();
    let header = t.write(&mut bytes).unwrap();
    let (back, read_header) = Table::read(bytes.as_slice()).unwrap();
    assert_eq!(back, t);
    assert_eq!(read_header, header);
    let mut corrupt = bytes.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(
        Table::read(corrupt.as_slice())
            .unwrap_err()
            .contains("digest")
    );
    assert!(Table::read(&bytes[..bytes.len() - 1]).is_err());
    assert!(Table::read(&bytes[..4]).is_err());
    assert!(Table::read(&b"NOTATABL"[..]).is_err());
}

#[test]
fn missing_lower_table_is_an_error() {
    let m = Material::parse("KQvK").unwrap();
    assert!(solve(&m, &[None, None], 2).unwrap_err().contains("lower"));
}
