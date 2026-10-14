use tablebase_core::pawnless::{
    Geometry, Material, Role, Solved, Table, king_reach, solve, solve_tree,
};

/// Knight moves of square `s` from file and rank steps, not from bit shifts.
fn knight_reach(s: u8) -> u64 {
    let (file, rank) = (i32::from(s % 8), i32::from(s / 8));
    let steps = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];
    steps
        .iter()
        .map(|(df, dr)| (file + df, rank + dr))
        .filter(|&(f, r)| (0..8).contains(&f) && (0..8).contains(&r))
        .fold(0, |bits, (f, r)| bits | 1 << (8 * r + f))
}

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
        assert_eq!(g.attacks(Role::Knight, s, 0), knight_reach(s));
        assert_eq!(g.king(s), king_reach(1 << s));
        let rook = g.attacks(Role::Rook, s, 0);
        assert_eq!(rook.count_ones(), 14);
        let queen = g.attacks(Role::Queen, s, 0);
        assert_eq!(queen, rook | g.attacks(Role::Bishop, s, 0));
    }
    // a1 rook blocked on a4 and d1. The between function excludes both endpoints.
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
    // verify_pawnless checked these counts against Syzygy (all legal positions).
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
    let header = t.write(&mut bytes, "test").unwrap();
    let (back, read_header) = Table::read(bytes.as_slice()).unwrap();
    assert_eq!(back, t);
    assert_eq!(read_header, header);
    assert_eq!(read_header.generator, "test");
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

#[test]
fn symmetries_agree_on_squares_and_bitboards() {
    use tablebase_core::pawnless::{Symmetry, canonical_white_king, symmetry_violations};
    assert_eq!(Symmetry::all().count(), 8);
    for sym in Symmetry::all() {
        let mut image = 0u64;
        for s in 0..64u8 {
            assert_eq!(sym.bitboard(1 << s), 1 << sym.square(s));
            image |= 1 << sym.square(s);
        }
        assert_eq!(image, u64::MAX, "{sym:?} is a permutation");
    }
    let triangle: Vec<u8> = (0..64).filter(|&s| canonical_white_king(s)).collect();
    assert_eq!(triangle, [0, 1, 2, 3, 9, 10, 11, 18, 19, 27]);
    for s in 0..64u8 {
        assert!(Symmetry::all().any(|sym| canonical_white_king(sym.square(s))));
    }
    let t = solved_kqvk();
    assert_eq!(symmetry_violations(&t, 4).unwrap(), 0);
    let mut broken = t.clone();
    broken.win[1_234] ^= 1 << 7;
    assert!(symmetry_violations(&broken, 4).unwrap() > 0);
}

/// The symmetry check also compares proof ranks, and positions that differ
/// only in the order of identical pieces.
#[test]
fn symmetry_check_covers_ranks_and_identical_pieces() {
    use tablebase_core::pawnless::{Solved, Symmetry, solve_tree, symmetry_violations};
    let m = Material::parse("KRRvK").unwrap();
    let mut solved = Solved::new();
    let mut cpu = |m: &Material, lower: &[Option<&Table>]| solve(m, lower, 8);
    let t = (*solve_tree(&m, &mut solved, &mut cpu).unwrap()).clone();
    assert_eq!(symmetry_violations(&t, 8).unwrap(), 0);
    let (w, bk) = (0..t.valid.len())
        .find_map(|w| {
            let (_, squares) = m.decode(w);
            let decisive = t.win[w] | t.loss[w];
            (squares[1] > squares[2] && decisive != 0).then(|| (w, decisive.trailing_zeros() as u8))
        })
        .unwrap();
    let mut one = t.clone();
    one.rank[w * 64 + usize::from(bk)] += 1;
    assert!(symmetry_violations(&one, 8).unwrap() > 0);
    // The same change on all eight images keeps the rooks out of order.
    let mut orbit = t.clone();
    let (black, squares) = m.decode(w);
    for sym in Symmetry::all() {
        let image: Vec<u8> = squares[..3].iter().map(|&s| sym.square(s)).collect();
        let v = m.encode(black, &image);
        orbit.rank[v * 64 + usize::from(sym.square(bk))] = t.rank[w * 64 + usize::from(bk)] + 1;
    }
    assert!(symmetry_violations(&orbit, 8).unwrap() > 0);
}

#[test]
fn format_v2_unranked_and_legacy_v1_files() {
    use sha2::{Digest, Sha256};
    let t = solved_kqvk();
    // Unranked (in-place) tables store 16-byte records.
    let mut unranked = t.clone();
    unranked.rank.clear();
    let mut bytes = Vec::new();
    let header = unranked.write(&mut bytes, "test").unwrap();
    assert_eq!(header.planes.len(), 2);
    assert_eq!(unranked.payload().len(), unranked.valid.len() * 16);
    assert_eq!(Table::read(bytes.as_slice()).unwrap().0, unranked);
    // A version 1 file: three planes then ranks, same header fields.
    let mut payload = Vec::new();
    for plane in [&t.valid, &t.win, &t.loss] {
        for w in plane.iter() {
            payload.extend_from_slice(&w.to_le_bytes());
        }
    }
    for r in &t.rank {
        payload.extend_from_slice(&r.to_le_bytes());
    }
    let json = serde_json::json!({
        "format": "tablebase-pawnless-v1", "material": "KQvK", "layout": "v1",
        "words": t.valid.len(), "rounds": t.rounds,
        "planes": ["valid:u64", "win:u64", "loss:u64", "rank:u16"],
        "payload_sha256": format!("{:x}", Sha256::digest(&payload)),
    })
    .to_string();
    let mut v1 = b"TBPAWNL1".to_vec();
    v1.extend_from_slice(&(json.len() as u64).to_le_bytes());
    v1.extend_from_slice(json.as_bytes());
    v1.extend_from_slice(&payload);
    assert_eq!(Table::read(v1.as_slice()).unwrap().0, t);
}

#[test]
fn rank_width_follows_round_count() {
    let mut t = solved_kqvk();
    assert_eq!(t.rank_width(), 1);
    let mut bytes = Vec::new();
    assert_eq!(t.write(&mut bytes, "test").unwrap().planes[2], "rank:u8x64");
    assert_eq!(Table::read(bytes.as_slice()).unwrap().0, t);
    // A table with 255 or more rounds stores u16 ranks.
    t.rounds = 300;
    t.rank[5 * 64 + 9] = 299;
    let mut bytes = Vec::new();
    assert_eq!(
        t.write(&mut bytes, "test").unwrap().planes[2],
        "rank:u16x64"
    );
    assert_eq!(Table::read(bytes.as_slice()).unwrap().0, t);
}
