use serde::Deserialize;
use tablebase_core::dense6::{Dense6, SLOTS, transform};
#[derive(Deserialize)]
struct Position {
    squares: [u8; 6],
    black: bool,
    id: String,
    canonical: [u8; 6],
    children: Vec<(usize, u8, String)>,
}
#[derive(Deserialize)]
struct Fixtures {
    maps: Vec<Option<(u16, u8)>>,
    positions: Vec<Position>,
    roundtrip_ids: Vec<String>,
}
#[test]
fn padded_index_and_legal_quiet_closure() {
    let fixtures: Fixtures = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/six_index.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let ranker = Dense6::new().unwrap();
    assert_eq!(fixtures.maps.len(), 4096);
    for (actual, expected) in ranker.king_map.iter().zip(fixtures.maps) {
        assert_eq!(actual.map(|x| (x.pair, x.symmetry)), expected);
    }
    for fixture in fixtures.positions {
        let id = fixture.id.parse().unwrap();
        assert_eq!(ranker.rank(fixture.squares, fixture.black).unwrap(), id);
        assert_eq!(
            ranker.unrank(id).unwrap(),
            (fixture.canonical, fixture.black)
        );
        for (axis, destination, child) in fixture.children {
            assert_eq!(
                ranker.relocate(id, axis, destination).unwrap(),
                child.parse::<u64>().unwrap()
            );
        }
    }
    for raw in fixtures.roundtrip_ids {
        let id = raw.parse().unwrap();
        let (squares, turn) = ranker.unrank(id).unwrap();
        assert_eq!(ranker.rank(squares, turn).unwrap(), id);
    }
    assert!(ranker.unrank(SLOTS).is_err());
    assert!(ranker.rank([0, 1, 2, 3, 4, 5], false).is_err());
    assert!(ranker.rank([0, 63, 64, 3, 4, 5], false).is_err());
    assert!(ranker.relocate(0, 6, 0).is_err());
    assert!(transform(64, 0).is_err());
    assert!(transform(0, 8).is_err());
}
