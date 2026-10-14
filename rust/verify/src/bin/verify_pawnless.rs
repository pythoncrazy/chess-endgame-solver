//! Independent check of solved pawnless tables, position by position.
//!
//! It uses shakmaty, not the move generator of the solver, for legality and legal
//! moves. It uses shakmaty-syzygy for the published WDL. For every position of every table:
//! - legality agrees with shakmaty. The check accepts impossible-check positions.
//! - the unclocked WDL equals the sign of Syzygy WDL after zeroing.
//! - proof ranks are exact. A lower-material child counts as rank 0:
//!   - a win or a loss has a proof rank.
//!   - the rank of a win is one more than the smallest rank of its child losses.
//!   - every child of a loss is a win. The rank of the loss is one more than the
//!     largest rank of its children.
//!   - a rank-0 loss is checkmate.
//!   - a draw has no child that is a loss. It is a stalemate or has a child that is not a win.
use shakmaty::{
    Board, CastlingMode, Chess, Color, FromSetup, Piece, Position, PositionError, Role, Setup,
    Square,
};
use shakmaty_syzygy::{
    Tablebase,
    filesystem::{Filesystem, RandomAccessFile, ReadHint},
};
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
use tablebase_core::packed::{Layout, load_table};
use tablebase_core::pawnless::{self, Material, Table, canonical_white_king};

type Tables = BTreeMap<String, Table>;

/// Syzygy files held in memory. Each probe is a slice copy instead of a
/// `pread` system call. Those calls dominated the profile of the verifier.
struct MemoryFilesystem;

struct MemoryFile(Vec<u8>);

impl Filesystem for MemoryFilesystem {
    fn regular_file_size(&self, path: &Path) -> io::Result<u64> {
        let meta = std::fs::metadata(path)?;
        if !meta.is_file() {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        Ok(meta.len())
    }

    fn read_dir(&self, path: &Path) -> io::Result<Vec<PathBuf>> {
        std::fs::read_dir(path)?
            .map(|e| e.map(|e| e.path()))
            .collect()
    }

    fn open(&self, path: &Path) -> io::Result<Box<dyn RandomAccessFile>> {
        Ok(Box::new(MemoryFile(std::fs::read(path)?)))
    }
}

impl RandomAccessFile for MemoryFile {
    fn read_at(&self, buf: &mut [u8], offset: u64, _: ReadHint) -> io::Result<usize> {
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(self.0.len());
        let n = buf.len().min(self.0.len() - start);
        buf[..n].copy_from_slice(&self.0[start..start + n]);
        Ok(n)
    }
}

/// A table and the lower tables its captures lead to, resolved one time.
struct Context<'a> {
    table: &'a Table,
    /// (captured piece, table without it) for every non-king skeleton piece.
    lower: Vec<(Piece, Option<&'a Table>)>,
}

impl<'a> Context<'a> {
    fn new(tables: &'a Tables, table: &'a Table) -> Context<'a> {
        let m = &table.material;
        let lower = (1..m.skeleton_len())
            .map(|k| {
                let below = m.without(k).ok().and_then(|l| tables.get(&l.name()));
                (piece(m.skeleton[k]), below)
            })
            .collect();
        Context { table, lower }
    }
}

/// Per-thread counts, added to the shared tally one time per chunk.
#[derive(Default)]
struct Counts {
    positions: u64,
    legal: u64,
    syzygy_compared: u64,
    proof_checked: u64,
}

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
/// None if two pieces share a square or if shakmaty rejects the position.
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

/// Find a board in a table of `material`: (word, black-king square).
/// Repeated pieces take their squares from low to high. Any assignment of
/// identical pieces is a valid duplicate cell of the working index.
fn locate(material: &Material, board: &Board, turn: Color) -> Option<(usize, u8)> {
    let mut used = 0u64;
    let mut squares = [0u8; pawnless::MAX_SKELETON];
    for (slot, p) in squares.iter_mut().zip(&material.skeleton) {
        let free = board.by_piece(piece(*p)).0 & !used;
        if free == 0 {
            return None;
        }
        *slot = free.trailing_zeros() as u8;
        used |= 1 << *slot;
    }
    let black_king = board.king_of(Color::Black)?;
    Some((
        material.encode(turn == Color::Black, &squares[..material.skeleton_len()]),
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

/// Value and rank of a child, which can be in a lower material.
fn child_value(
    ctx: &Context,
    child: &Chess,
    captured: Option<Role>,
) -> Result<(i8, Option<u16>), String> {
    let (table, same) = match captured {
        None => (ctx.table, true),
        Some(role) => {
            let captured = Piece {
                color: child.turn(),
                role,
            };
            let (_, below) = ctx
                .lower
                .iter()
                .find(|(p, _)| *p == captured)
                .ok_or("captured piece absent from material")?;
            (below.ok_or("missing lower table")?, false)
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
    ctx: &Context,
    syzygy: &Tablebase<Chess>,
    word: usize,
    bk: u8,
    tally: &mut Counts,
) -> Result<(), String> {
    let table = ctx.table;
    let material = &table.material;
    let solved = table.wdl(word, bk);
    let Some(pos) = position(material, word, bk) else {
        return match solved {
            None => Ok(()),
            Some(_) => Err("solver marks a shakmaty-illegal position legal".into()),
        };
    };
    let value = solved.ok_or("solver marks a legal position illegal")?;
    tally.legal += 1;
    if material.skeleton_len() > 1 {
        let wdl = syzygy
            .probe_wdl_after_zeroing(&pos)
            .map_err(|e| format!("syzygy: {e}"))? as i8;
        if value != 2 * wdl.signum() {
            return Err(format!("syzygy WDL {wdl} vs solved {value}"));
        }
        tally.syzygy_compared += 1;
    } else if value != 0 {
        return Err("bare kings must be a draw".into());
    }
    if table.rank.is_empty() {
        // Solved in place: there are no proof ranks to check. Check legality and WDL only.
        return Ok(());
    }
    let rank = table.rank[word * 64 + usize::from(bk)];
    if value != 0 && rank == pawnless::UNRESOLVED {
        return Err(format!("value {value} has no proof rank"));
    }
    let moves = pos.legal_moves();
    // The rank is the solver round. A win is one more than its fastest child
    // that is a loss. A loss is one more than its slowest child. A child in a
    // lower material counts as 0. Thus a rank is the distance to mate in
    // plies, if no capture is on the path. Over all children, find the fastest
    // loss, the slowest child, and any child that is not a win. The loop stops
    // when the obligation fails. The stored image of a skipped child and the
    // symmetry check cover its legality. A checkmate (rank 0) has no child.
    let rank = u32::from(rank);
    let (mut fastest_loss, mut slowest, mut any_nonwin) = (None::<u32>, 0u32, false);
    let probe_children = !(value == -2 && rank == 0);
    for m in moves.iter().filter(|_| probe_children) {
        let mut child = pos.clone();
        child.play_unchecked(*m);
        let (v, r) = child_value(ctx, &child, m.capture())?;
        let r = u32::from(r.unwrap_or(0));
        if v == -2 {
            fastest_loss = Some(fastest_loss.map_or(r, |f| f.min(r)));
        }
        slowest = slowest.max(r);
        any_nonwin |= v != 2;
        let failed = match value {
            2 => fastest_loss.is_some_and(|f| f + 1 < rank),
            -2 => any_nonwin || slowest + 1 > rank,
            _ => fastest_loss.is_some(),
        };
        if failed {
            break;
        }
    }
    let ok = match value {
        2 => fastest_loss.is_some_and(|f| f + 1 == rank),
        -2 if rank == 0 => moves.is_empty() && pos.is_check(),
        -2 => !moves.is_empty() && !any_nonwin && slowest + 1 == rank,
        _ => {
            rank == u32::from(pawnless::UNRESOLVED)
                && fastest_loss.is_none()
                && (if moves.is_empty() {
                    !pos.is_check()
                } else {
                    any_nonwin
                })
        }
    };
    if !ok {
        return Err(format!(
            "proof obligation fails (value {value}, rank {rank})"
        ));
    }
    tally.proof_checked += 1;
    Ok(())
}

fn record_failure(
    table: &Table,
    word: usize,
    bk: u8,
    error: &str,
    tally: &Tally,
    first_failure: &std::sync::Mutex<Option<String>>,
) {
    tally.failures.fetch_add(1, Ordering::Relaxed);
    let mut slot = first_failure.lock().expect("mutex");
    if slot.is_none() {
        let fen = position(&table.material, word, bk).map(|p| {
            shakmaty::fen::Fen::from_position(&p, shakmaty::EnPassantMode::Legal).to_string()
        });
        *slot = Some(format!("word {word} bk {bk} fen {fen:?}: {error}"));
    }
}

/// Material of a table file named `MATERIAL.*` (v1/v2 `.gpu.tbp`, v3 `.tbz`).
fn material_of(path: &str) -> Result<Material, String> {
    let name = std::path::Path::new(path)
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.split('.').next())
        .ok_or(format!("{path}: no material in the file name"))?;
    Material::parse(name)
}

/// The material and the lower materials its captures lead to.
fn needs(m: &Material) -> Vec<String> {
    let mut out = vec![m.name()];
    out.extend((1..m.skeleton_len()).filter_map(|k| m.without(k).ok().map(|l| l.name())));
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    All,
    /// White king in the a1-d1-d4 triangle, plus the symmetry check.
    Canonical,
    /// One position per symmetry class (the v3 stored set), plus the
    /// symmetry check.
    Stored,
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: verify_pawnless SYZYGY_DIR[,DIR...] THREADS [--canonical|--stored] \
                 TABLE_FILE...";
    let syzygy_dirs = args.next().ok_or(usage)?;
    let threads: usize = args
        .next()
        .ok_or(usage)?
        .parse()
        .map_err(|e| format!("threads: {e}"))?;
    let mut args = args.peekable();
    let scope = match args.next_if(|a| a.starts_with("--")).as_deref() {
        None => Scope::All,
        Some("--canonical") => Scope::Canonical,
        Some("--stored") => Scope::Stored,
        Some(_) => return Err(usage.into()),
    };
    let mut paths = BTreeMap::new();
    for path in args {
        let name = material_of(&path)?.name();
        if let Some(other) = paths.insert(name.clone(), path) {
            return Err(format!("{other}: a second file has the material {name}"));
        }
    }
    if paths.is_empty() {
        return Err(usage.into());
    }
    let mut order: Vec<Material> = paths
        .keys()
        .map(|n| Material::parse(n))
        .collect::<Result<_, _>>()?;
    order.sort_by_key(|m| (m.skeleton_len(), m.name()));
    let mut syzygy = Tablebase::with_filesystem(Arc::new(MemoryFilesystem));
    for dir in syzygy_dirs.split(',') {
        syzygy
            .add_directory(dir)
            .map_err(|e| format!("{dir}: {e}"))?;
    }
    // Tables stay loaded only while a material that is not done yet reads them.
    let mut tables = Tables::new();
    let mut digests = BTreeMap::new();
    let mut report = Vec::new();
    for (i, material) in order.iter().enumerate() {
        for name in needs(material) {
            if let (false, Some(path)) = (tables.contains_key(&name), paths.get(&name)) {
                let (table, digest) = load_table(path.as_ref(), threads)?;
                if table.material.name() != name {
                    return Err(format!(
                        "{path}: the file has the material {}",
                        table.material.name()
                    ));
                }
                digests.insert(name.clone(), digest);
                tables.insert(name, table);
            }
        }
        let mut row = verify_table(&tables, &tables[&material.name()], &syzygy, scope, threads)?;
        row["payload_sha256"] = digests[&material.name()].clone().into();
        eprintln!("{row}");
        report.push(row);
        let keep: std::collections::BTreeSet<String> =
            order[i + 1..].iter().flat_map(needs).collect();
        tables.retain(|name, _| keep.contains(name));
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

fn verify_table(
    tables: &Tables,
    table: &Table,
    syzygy: &Tablebase<Chess>,
    scope: Scope,
    threads: usize,
) -> Result<serde_json::Value, String> {
    let start = Instant::now();
    let tally = Tally::default();
    let first_failure = std::sync::Mutex::new(None::<String>);
    let layout = Layout::new(&table.material);
    // Stored scope walks only the stored words (white king in the triangle).
    let words = match scope {
        Scope::Stored => layout.words(),
        _ => table.valid.len(),
    };
    let ctx = Context::new(tables, table);
    // Dynamic chunks: canonical words sit in contiguous index ranges, so a
    // static split would leave most threads idle.
    const CHUNK: usize = 4096;
    let next = AtomicU64::new(0);
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            let (layout, ctx, tally, first_failure, next) =
                (&layout, &ctx, &tally, &first_failure, &next);
            s.spawn(move || {
                loop {
                    let start = next.fetch_add(CHUNK as u64, Ordering::Relaxed) as usize;
                    if start >= words {
                        break;
                    }
                    let mut counts = Counts::default();
                    for index in start..(start + CHUNK).min(words) {
                        let word = match scope {
                            Scope::Stored => layout.full_word(index),
                            _ => index,
                        };
                        let mut squares = match scope {
                            Scope::All => u64::MAX,
                            Scope::Canonical => {
                                if canonical_white_king(white_king(&table.material, word)) {
                                    u64::MAX
                                } else {
                                    0
                                }
                            }
                            Scope::Stored => layout.stored_in_word(word),
                        };
                        while squares != 0 {
                            let bk = squares.trailing_zeros() as u8;
                            squares &= squares - 1;
                            counts.positions += 1;
                            if let Err(e) = check_position(ctx, syzygy, word, bk, &mut counts) {
                                record_failure(table, word, bk, &e, tally, first_failure);
                            }
                        }
                    }
                    tally
                        .positions
                        .fetch_add(counts.positions, Ordering::Relaxed);
                    tally.legal.fetch_add(counts.legal, Ordering::Relaxed);
                    let compared = counts.syzygy_compared;
                    tally.syzygy_compared.fetch_add(compared, Ordering::Relaxed);
                    let checked = counts.proof_checked;
                    tally.proof_checked.fetch_add(checked, Ordering::Relaxed);
                }
            });
        }
    });
    let symmetry = (scope != Scope::All)
        .then(|| pawnless::symmetry_violations(table, threads))
        .transpose()?;
    let failures = tally.failures.load(Ordering::Relaxed) + symmetry.unwrap_or(0);
    Ok(serde_json::json!({
        "material": table.material.name(),
        "positions_examined": tally.positions.load(Ordering::Relaxed),
        "legal_positions": tally.legal.load(Ordering::Relaxed),
        "syzygy_wdl_compared": tally.syzygy_compared.load(Ordering::Relaxed),
        "proof_obligations_checked": tally.proof_checked.load(Ordering::Relaxed),
        "scope": match scope {
            Scope::All => "all",
            Scope::Canonical => "canonical-white-king-triangle-plus-symmetry",
            Scope::Stored => "one-position-per-symmetry-class-plus-symmetry",
        },
        "symmetry_violating_words": symmetry,
        "proof_ranks_present": !table.rank.is_empty(),
        "failures": failures,
        "first_failure": first_failure.into_inner().expect("mutex"),
        "seconds": start.elapsed().as_secs_f64(),
        "status": if failures == 0 { "pass" } else { "FAIL" },
    }))
}
