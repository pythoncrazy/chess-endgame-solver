//! Independent check of solved pawnless tables, position by position.
//!
//! Uses shakmaty (not the solver's move generator) for legality and legal moves,
//! and shakmaty-syzygy for published WDL. For every position of every table:
//! - legality agrees with shakmaty (impossible-check positions are accepted);
//! - the unclocked WDL equals the sign of Syzygy WDL after zeroing;
//! - proof ranks hold: a win has a child loss of smaller rank (or a capture
//!   into a lower-material loss), every child of a loss is a win of smaller
//!   rank (or a lower-material win), a rank-0 loss is checkmate, and a draw
//!   has no losing child and is a stalemate or has a nonwinning child.
use shakmaty::{
    Board, CastlingMode, Chess, Color, FromSetup, Piece, Position, PositionError, Role, Setup,
    Square,
};
use shakmaty_syzygy::Tablebase;
use std::{
    collections::BTreeMap,
    fs::File,
    io::BufReader,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};
use tablebase_core::pawnless::{self, Material, Table, canonical_white_king};

type Tables = BTreeMap<String, Table>;

fn white_king(material: &Material, word: usize) -> u8 {
    material.decode(word).1[0]
}

fn role(role: pawnless::Role) -> Role {
    match role {
        pawnless::Role::King => Role::King,
        pawnless::Role::Queen => Role::Queen,
        pawnless::Role::Rook => Role::Rook,
        pawnless::Role::Bishop => Role::Bishop,
        pawnless::Role::Knight => Role::Knight,
    }
}

fn piece(p: pawnless::Piece) -> Piece {
    Piece {
        color: Color::from_white(p.white),
        role: role(p.role),
    }
}

/// Build the position of `word` with the black king on `black_king`. Returns
/// None for overlapping pieces or a position shakmaty rejects.
fn position(material: &Material, word: usize, black_king: u8) -> Option<Chess> {
    let (black_to_move, squares) = material.decode(word);
    let mut board = Board::empty();
    let mut occupied = 1u64 << black_king;
    for (p, &square) in material.skeleton.iter().zip(&squares) {
        if occupied >> square & 1 != 0 {
            return None;
        }
        occupied |= 1 << square;
        board.set_piece_at(Square::new(u32::from(square)), piece(*p));
    }
    board.set_piece_at(
        Square::new(u32::from(black_king)),
        Piece {
            color: Color::Black,
            role: Role::King,
        },
    );
    let mut setup = Setup::empty();
    setup.board = board;
    setup.turn = Color::from_white(!black_to_move);
    Chess::from_setup(setup, CastlingMode::Standard)
        .or_else(PositionError::ignore_impossible_check)
        .ok()
}

/// Locate a board in a table of `material`: (word, black-king square).
/// Repeated pieces take their squares in increasing order; any assignment of
/// identical pieces is a valid duplicate cell of the working index.
fn locate(material: &Material, board: &Board, turn: Color) -> Option<(usize, u8)> {
    let mut used = 0u64;
    let mut squares = Vec::with_capacity(material.skeleton_len());
    for p in &material.skeleton {
        let free = board.by_piece(piece(*p)).0 & !used;
        if free == 0 {
            return None;
        }
        let square = free.trailing_zeros() as u8;
        used |= 1 << square;
        squares.push(square);
    }
    let black_king = board.king_of(Color::Black)?;
    Some((
        material.encode(turn == Color::Black, &squares),
        u32::from(black_king) as u8,
    ))
}

#[derive(Default)]
struct Tally {
    positions: AtomicU64,
    legal: AtomicU64,
    syzygy_compared: AtomicU64,
    proof_checked: AtomicU64,
    failures: AtomicU64,
}

/// Value and rank of a child, which may be in a lower material.
fn child_value(
    tables: &Tables,
    material: &Material,
    child: &Chess,
    captured: Option<Role>,
) -> Result<(i8, Option<u16>), String> {
    let (table, same) = match captured {
        None => (&tables[&material.name()], true),
        Some(r) => {
            let color = child.turn();
            let k = (1..material.skeleton_len())
                .find(|&k| piece(material.skeleton[k]) == Piece { color, role: r })
                .ok_or("captured piece absent from material")?;
            let name = material.without(k)?.name();
            (
                tables.get(&name).ok_or(format!("missing table {name}"))?,
                false,
            )
        }
    };
    let (word, bk) =
        locate(&table.material, child.board(), child.turn()).ok_or("child cannot be located")?;
    let value = table
        .wdl(word, bk)
        .ok_or("legal child is invalid in table")?;
    let rank = table.rank.get(word * 64 + usize::from(bk)).copied();
    Ok((value, if same { rank } else { None }))
}

fn check_position(
    tables: &Tables,
    syzygy: &Tablebase<Chess>,
    table: &Table,
    word: usize,
    bk: u8,
    tally: &Tally,
) -> Result<(), String> {
    let material = &table.material;
    let solved = table.wdl(word, bk);
    let Some(pos) = position(material, word, bk) else {
        return match solved {
            None => Ok(()),
            Some(_) => Err("solver marks a shakmaty-illegal position legal".into()),
        };
    };
    let value = solved.ok_or("solver marks a legal position illegal")?;
    tally.legal.fetch_add(1, Ordering::Relaxed);
    if material.skeleton_len() > 1 {
        let wdl = syzygy
            .probe_wdl_after_zeroing(&pos)
            .map_err(|e| format!("syzygy: {e}"))? as i8;
        if value != 2 * wdl.signum() {
            return Err(format!("syzygy WDL {wdl} vs solved {value}"));
        }
        tally.syzygy_compared.fetch_add(1, Ordering::Relaxed);
    } else if value != 0 {
        return Err("bare kings must be a draw".into());
    }
    if table.rank.is_empty() {
        // Solved in place: no proof ranks to check; legality and WDL only.
        return Ok(());
    }
    let rank = table.rank[word * 64 + usize::from(bk)];
    let moves = pos.legal_moves();
    let mut children = Vec::with_capacity(moves.len());
    for m in &moves {
        let mut child = pos.clone();
        child.play_unchecked(*m);
        children.push(child_value(tables, material, &child, m.capture())?);
    }
    let smaller = |r: Option<u16>| r.is_none_or(|r| r < rank);
    let ok = match value {
        2 => children.iter().any(|&(v, r)| v == -2 && smaller(r)),
        -2 if rank == 0 => moves.is_empty() && pos.is_check(),
        -2 => !moves.is_empty() && children.iter().all(|&(v, r)| v == 2 && smaller(r)),
        _ => {
            rank == pawnless::UNRESOLVED
                && !children.iter().any(|&(v, _)| v == -2)
                && (if moves.is_empty() {
                    !pos.is_check()
                } else {
                    children.iter().any(|&(v, _)| v != 2)
                })
        }
    };
    if !ok {
        return Err(format!(
            "proof obligation fails (value {value}, rank {rank})"
        ));
    }
    tally.proof_checked.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: verify_pawnless SYZYGY_DIRECTORY THREADS [--canonical] TABLE_FILE...";
    let syzygy_dir = args.next().ok_or(usage)?;
    let threads: usize = args
        .next()
        .ok_or(usage)?
        .parse()
        .map_err(|e| format!("threads: {e}"))?;
    let mut args = args.peekable();
    // Canonical: check positions with the white king in the a1-d1-d4 triangle,
    // and check that each table is invariant under the eight symmetries.
    let canonical = args.next_if(|a| a == "--canonical").is_some();
    let mut tables = Tables::new();
    let mut digests = BTreeMap::new();
    for path in args {
        let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        let (table, header) = Table::read(BufReader::new(file))?;
        digests.insert(header.material.clone(), header.payload_sha256);
        tables.insert(header.material, table);
    }
    if tables.is_empty() {
        return Err(usage.into());
    }
    let mut syzygy = Tablebase::new();
    syzygy
        .add_directory(&syzygy_dir)
        .map_err(|e| format!("{syzygy_dir}: {e}"))?;
    let mut report = Vec::new();
    for table in tables.values() {
        let start = Instant::now();
        let tally = Tally::default();
        let first_failure = std::sync::Mutex::new(None::<String>);
        let words = table.valid.len();
        let chunk = words.div_ceil(threads.max(1));
        std::thread::scope(|scope| {
            for part in 0..threads.max(1) {
                let (tables, syzygy, tally, first_failure) =
                    (&tables, &syzygy, &tally, &first_failure);
                scope.spawn(move || {
                    for word in part * chunk..((part + 1) * chunk).min(words) {
                        if canonical && !canonical_white_king(white_king(&table.material, word)) {
                            continue;
                        }
                        for bk in 0..64u8 {
                            tally.positions.fetch_add(1, Ordering::Relaxed);
                            if let Err(e) = check_position(tables, syzygy, table, word, bk, tally) {
                                tally.failures.fetch_add(1, Ordering::Relaxed);
                                let mut slot = first_failure.lock().expect("mutex");
                                if slot.is_none() {
                                    let fen = position(&table.material, word, bk).map(|p| {
                                        shakmaty::fen::Fen::from_position(
                                            &p,
                                            shakmaty::EnPassantMode::Legal,
                                        )
                                        .to_string()
                                    });
                                    *slot = Some(format!("word {word} bk {bk} fen {fen:?}: {e}"));
                                }
                            }
                        }
                    }
                });
            }
        });
        let symmetry = canonical
            .then(|| pawnless::symmetry_violations(table, threads))
            .transpose()?;
        let failures = tally.failures.load(Ordering::Relaxed) + symmetry.unwrap_or(0);
        report.push(serde_json::json!({
            "material": table.material.name(),
            "payload_sha256": digests[&table.material.name()],
            "positions_examined": tally.positions.load(Ordering::Relaxed),
            "legal_positions": tally.legal.load(Ordering::Relaxed),
            "syzygy_wdl_compared": tally.syzygy_compared.load(Ordering::Relaxed),
            "proof_obligations_checked": tally.proof_checked.load(Ordering::Relaxed),
            "scope": if canonical { "canonical-white-king-triangle-plus-symmetry" } else { "all" },
            "symmetry_violating_words": symmetry,
            "proof_ranks_present": !table.rank.is_empty(),
            "failures": failures,
            "first_failure": first_failure.into_inner().expect("mutex"),
            "seconds": start.elapsed().as_secs_f64(),
            "status": if failures == 0 { "pass" } else { "FAIL" },
        }));
    }
    let failed = report.iter().any(|r| r["status"] != "pass");
    println!(
        "{}",
        serde_json::json!({"verifier": "shakmaty+syzygy", "tables": report})
    );
    if failed {
        return Err("verification failed".into());
    }
    Ok(())
}
