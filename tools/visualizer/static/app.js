// Tablebase analysis board. The client generates pawnless chess moves itself
// (no pawns, castling or en passant), so the server sends only the value of each
// legal move, in sorted UCI order. Responses are cached per FEN.

const $ = (id) => document.getElementById(id);
const FILES = "abcdefgh";
const SIDE = { w: "White", b: "Black" };
const START = "8/8/8/4k3/8/8/8/KQ6 w - - 0 1";
const SVG = "http://www.w3.org/2000/svg";

// ------------------------------------------------------------ chess core (pawnless)
const ORTHO = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const DIAG = [[1, 1], [-1, 1], [1, -1], [-1, -1]];
const MOVES = {
  N: { dirs: [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]], slide: false },
  K: { dirs: [...ORTHO, ...DIAG], slide: false },
  R: { dirs: ORTHO, slide: true },
  B: { dirs: DIAG, slide: true },
  Q: { dirs: [...ORTHO, ...DIAG], slide: true },
};

/** Squares a piece on `s` attacks (rays stop at, and include, the first piece). */
function attacks(board, s) {
  const { dirs, slide } = MOVES[board[s][1]];
  const out = [];
  for (const [df, dr] of dirs) {
    let f = s % 8, r = s >> 3;
    for (;;) {
      f += df; r += dr;
      if (f < 0 || f > 7 || r < 0 || r > 7) break;
      const t = r * 8 + f;
      out.push(t);
      if (!slide || board[t]) break;
    }
  }
  return out;
}

function attacked(board, sq, by) {
  for (let s = 0; s < 64; s++) {
    if (board[s]?.[0] === by && attacks(board, s).includes(sq)) return true;
  }
  return false;
}

const other = (c) => (c === "w" ? "b" : "w");
const kingSq = (board, c) => board.indexOf(`${c}K`);
const inCheck = (board, c) => attacked(board, kingSq(board, c), other(c));

function play(board, from, to) {
  const next = board.slice();
  next[to] = next[from];
  next[from] = null;
  return next;
}

function legalMoves(board, turn) {
  const moves = [];
  for (let s = 0; s < 64; s++) {
    if (board[s]?.[0] !== turn) continue;
    for (const t of attacks(board, s)) {
      if (board[t]?.[0] === turn || board[t]?.[1] === "K") continue;
      const next = play(board, s, t);
      if (!inCheck(next, turn)) moves.push({ from: s, to: t, uci: sqName(s) + sqName(t), capture: !!board[t], next });
    }
  }
  return moves.sort((a, b) => (a.uci < b.uci ? -1 : 1));
}

function san(board, turn, m, all) {
  const role = board[m.from][1];
  const rivals = all.filter((o) => o.to === m.to && o.from !== m.from && board[o.from][1] === role);
  let dis = "";
  if (rivals.length) {
    const sameFile = rivals.some((o) => o.from % 8 === m.from % 8);
    const sameRank = rivals.some((o) => o.from >> 3 === m.from >> 3);
    dis = !sameFile ? FILES[m.from % 8] : !sameRank ? String((m.from >> 3) + 1) : sqName(m.from);
  }
  const opp = other(turn);
  const check = inCheck(m.next, opp);
  const mate = check && legalMoves(m.next, opp).length === 0;
  return `${role}${dis}${m.capture ? "x" : ""}${sqName(m.to)}${mate ? "#" : check ? "+" : ""}`;
}

// ------------------------------------------------------------ FEN
function parseFen(fen) {
  const [placement, turn] = fen.trim().split(/\s+/);
  const rows = placement?.split("/");
  if (!rows || rows.length !== 8 || !["w", "b"].includes(turn)) return null;
  const board = new Array(64).fill(null);
  for (let r = 0; r < 8; r++) {
    let f = 0;
    for (const ch of rows[r]) {
      if (/[1-8]/.test(ch)) { f += Number(ch); continue; }
      if (!/[kqrbnpKQRBNP]/.test(ch) || f > 7) return null;
      board[(7 - r) * 8 + f] = (ch === ch.toUpperCase() ? "w" : "b") + ch.toUpperCase();
      f += 1;
    }
    if (f !== 8) return null;
  }
  return { board, turn };
}

function toFen(board, turn) {
  const rows = [];
  for (let r = 7; r >= 0; r--) {
    let row = "", empty = 0;
    for (let f = 0; f < 8; f++) {
      const p = board[r * 8 + f];
      if (!p) { empty++; continue; }
      if (empty) { row += empty; empty = 0; }
      row += p[0] === "w" ? p[1] : p[1].toLowerCase();
    }
    rows.push(row + (empty || ""));
  }
  return `${rows.join("/")} ${turn} - - 0 1`;
}

const sqName = (s) => FILES[s % 8] + ((s >> 3) + 1);
const sqIndex = (n) => FILES.indexOf(n[0]) + 8 * (Number(n[1]) - 1);

// ------------------------------------------------------------ state
const state = {
  board: new Array(64).fill(null),
  turn: "w",
  flipped: false,
  tool: "move",
  selected: null,
  moves: [], // legal moves of the current position (client-generated)
  result: null, // server reply for the current position
  nodes: [], // [{ fen, uci, san }], index 0 is the setup
  index: 0,
  request: 0,
};
const cache = new Map();
let materials = null; // { name: row }

function probe(fen) {
  if (!cache.has(fen)) {
    cache.set(fen, fetch(`/api/probe?fen=${encodeURIComponent(fen)}`)
      .then((r) => r.json())
      .catch(() => ({ s: "offline" })));
    if (cache.size > 5000) cache.delete(cache.keys().next().value);
  }
  return cache.get(fen);
}

async function show(fen, { push = null, reset = false } = {}) {
  const parsed = parseFen(fen);
  if (!parsed) { $("fen").classList.add("is-bad"); return; }
  $("fen").classList.remove("is-bad");
  Object.assign(state, { board: parsed.board, turn: parsed.turn, selected: null });
  if (push) {
    state.nodes = state.nodes.slice(0, state.index + 1);
    state.nodes.push({ fen, ...push });
    state.index = state.nodes.length - 1;
  } else if (reset) {
    state.nodes = [{ fen }];
    state.index = 0;
  }
  const pawnless = !state.board.some((p) => p?.[1] === "P");
  const kings = state.board.filter((p) => p === "wK").length === 1 && state.board.filter((p) => p === "bK").length === 1;
  state.moves = pawnless && kings ? legalMoves(state.board, state.turn) : [];
  state.result = null;
  history.replaceState(null, "", `${location.search}#${encodeURIComponent(fen)}`);
  $("fen").value = fen;
  renderBoard();
  renderTview();
  renderTurn();
  const id = ++state.request;
  const result = await probe(fen);
  if (id !== state.request) return;
  state.result = result;
  renderBoard();
  renderTools();
  renderSide();
}

const edit = () => show(toFen(state.board, state.turn), { reset: true });

// ------------------------------------------------------------ board
function pieceSvg(code) {
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("class", "pc");
  const use = document.createElementNS(SVG, "use");
  use.setAttribute("href", `#${code}`);
  svg.append(use);
  return svg;
}

function renderBoard() {
  const node = state.nodes[state.index];
  const last = node?.uci ? [sqIndex(node.uci.slice(0, 2)), sqIndex(node.uci.slice(2, 4))] : [];
  const dests = new Set(state.selected === null ? [] : state.moves.filter((m) => m.from === state.selected).map((m) => m.to));
  const check = checkSquare();
  const frag = document.createDocumentFragment();
  for (let i = 0; i < 64; i++) {
    const row = i >> 3, col = i % 8;
    const sq = state.flipped ? row * 8 + (7 - col) : (7 - row) * 8 + col;
    const cell = document.createElement("div");
    cell.className = `sq ${((sq >> 3) + (sq % 8)) % 2 ? "l" : "d"}`;
    cell.dataset.sq = sq;
    const p = state.board[sq];
    cell.setAttribute("role", "gridcell");
    cell.setAttribute("aria-label", sqName(sq) + (p ? ` ${SIDE[p[0]]} ${p[1]}` : ""));
    if (last.includes(sq)) cell.classList.add("last");
    if (state.selected === sq) cell.classList.add("sel");
    if (dests.has(sq)) cell.classList.add("dest", ...(p ? ["occ"] : []));
    if (sq === check) cell.classList.add("check");
    if (p) cell.append(pieceSvg(p));
    if (col === 0) cell.append(coord("r", String((sq >> 3) + 1)));
    if (row === 7) cell.append(coord("f", FILES[sq % 8]));
    frag.append(cell);
  }
  $("board").replaceChildren(frag);
}

function checkSquare() {
  const k = kingSq(state.board, state.turn);
  if (k < 0 || kingSq(state.board, other(state.turn)) < 0) return -1;
  return inCheck(state.board, state.turn) ? k : -1;
}

function coord(kind, text) {
  const s = document.createElement("span");
  s.className = `coord ${kind}`;
  s.textContent = text;
  return s;
}

function squareAt(x, y) {
  const cell = document.elementFromPoint(x, y)?.closest(".sq");
  return cell && $("board").contains(cell) ? Number(cell.dataset.sq) : null;
}

const moveFor = (from, to) => state.moves.find((m) => m.from === from && m.to === to);

function playMove(m) {
  show(toFen(m.next, other(state.turn)), { push: { uci: m.uci, san: san(state.board, state.turn, m, state.moves) } });
}

let drag = null;
$("board").addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  const sq = squareAt(e.clientX, e.clientY);
  if (sq === null) return;
  const p = state.board[sq];
  if (state.tool === "trash") { if (p) { state.board[sq] = null; edit(); } return; }
  if (state.tool !== "move") { state.board[sq] = p === state.tool ? null : state.tool; edit(); return; }
  if (state.selected !== null && state.selected !== sq) {
    const m = moveFor(state.selected, sq);
    if (m) { playMove(m); return; }
  }
  state.selected = p && p[0] === state.turn ? sq : null;
  renderBoard();
  if (!p) return;
  const size = $("board").clientWidth / 8;
  const ghost = pieceSvg(p);
  ghost.classList.add("drag");
  ghost.style.width = ghost.style.height = `${size}px`;
  drag = { from: sq, piece: p, ghost, size, moved: false, x: e.clientX, y: e.clientY };
  $("board").setPointerCapture(e.pointerId);
});

$("board").addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 4) return;
  if (!drag.moved) {
    drag.moved = true;
    document.body.append(drag.ghost);
    $("board").querySelector(`[data-sq="${drag.from}"] .pc`)?.classList.add("ghost");
  }
  drag.ghost.style.left = `${e.clientX - drag.size / 2}px`;
  drag.ghost.style.top = `${e.clientY - drag.size / 2}px`;
});

$("board").addEventListener("pointerup", (e) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  d.ghost.remove();
  if (!d.moved) return;
  const to = squareAt(e.clientX, e.clientY);
  if (to === d.from) { renderBoard(); return; }
  const m = to === null ? null : moveFor(d.from, to);
  if (m) { playMove(m); return; }
  // Not a legal move: the drop edits the position; off the board removes the piece.
  state.board[d.from] = null;
  if (to !== null) state.board[to] = d.piece;
  edit();
});

$("board").addEventListener("contextmenu", (e) => {
  e.preventDefault();
  const sq = squareAt(e.clientX, e.clientY);
  if (sq !== null && state.board[sq]) { state.board[sq] = null; edit(); }
});

// ------------------------------------------------------------ editor tray
const ROLE_NAME = { K: "king", Q: "queen", R: "rook", B: "bishop", N: "knight" };

function renderTray() {
  for (const row of document.querySelectorAll(".tray-row")) {
    const color = row.dataset.color;
    row.replaceChildren();
    for (const r of "KQRBN") {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "tray-btn";
      b.dataset.tool = color + r;
      b.title = `Place ${SIDE[color].toLowerCase()} ${ROLE_NAME[r]}`;
      b.setAttribute("aria-label", b.title);
      b.append(pieceSvg(color + r));
      row.append(b);
    }
    if (color === "w") {
      row.append(toolButton("move", "Move pieces (drag off the board to remove)", '<path d="M5 3l14 8-6 2-2 6z"/>'));
      row.append(toolButton("trash", "Remove pieces", '<path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13"/>'));
    } else {
      const hint = document.createElement("span");
      hint.className = "tray-hint";
      hint.textContent = "Pawnless · up to 6 pieces · right-click removes";
      row.append(hint);
    }
  }
  syncTray();
}

function toolButton(tool, title, path) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "tray-btn tool";
  b.dataset.tool = tool;
  b.title = title;
  b.setAttribute("aria-label", title);
  b.innerHTML = `<svg viewBox="0 0 24 24">${path}</svg>`;
  return b;
}

function syncTray() {
  for (const b of document.querySelectorAll(".tray-btn")) {
    const on = b.dataset.tool === state.tool;
    b.classList.toggle("is-on", on);
    b.setAttribute("aria-pressed", String(on));
  }
}

$("tray").addEventListener("click", (e) => {
  const b = e.target.closest(".tray-btn");
  if (!b) return;
  state.tool = state.tool === b.dataset.tool ? "move" : b.dataset.tool;
  state.selected = null;
  syncTray();
  renderBoard();
});

// ------------------------------------------------------------ tools column
const RES = { 2: "Win", 0: "Draw", "-2": "Loss" };
const cls = (w) => (w === 2 ? "win" : w === -2 ? "loss" : w === 0 ? "draw" : "unknown");
const SYZ = { 2: "Win", 1: "Cursed win", 0: "Draw", "-1": "Blessed loss", "-2": "Loss" };
const OK = '<svg class="mark" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 8.5 3 3 7-7"/></svg>';
const BAD = '<svg class="mark" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 4 8 8m0-8-8 8"/></svg>';
const WHY = {
  pawns: "Pawns are outside the pawnless tables.",
  castling: "Castling rights are not represented.",
  kings: "Each side needs exactly one king.",
  pieces: "The tables cover at most six pieces.",
};
const TITLES = {
  illegal: "Illegal position",
  unsupported: "Outside the tables",
  missing: "No table for this material",
  fen: "Invalid FEN",
  offline: "Server unavailable",
};

/** Moves joined with their server values; null when the reply does not match. */
function valuedMoves(r) {
  if (r?.s !== "ok" || r.n !== state.moves.length) return null;
  const disagree = new Set(r.d ?? []);
  const opp = other(state.turn);
  return state.moves.map((m, i) => {
    const [w, rank] = r.v[i];
    const none = legalMoves(m.next, opp).length === 0;
    const check = inCheck(m.next, opp);
    return { ...m, w, rank, mate: none && check, stale: none && !check, disagree: disagree.has(i) };
  });
}

function renderTools() {
  const r = state.result;
  const exact = materials?.[r?.m]?.exactDtm;
  const pearl = $("pearl"), sub = $("ceval-sub"), gauge = $("gauge");
  clearArrows();
  if (r?.s !== "ok") {
    pearl.textContent = "–";
    pearl.className = "pearl";
    gauge.classList.add("is-off");
    const title = TITLES[r?.s] ?? "No result";
    const detail = r?.s === "unsupported" ? WHY[r.why] : r?.s === "missing" ? `${r.m} has not been solved.` : r?.s === "illegal" ? "The side not to move is in check, or the kings touch." : "";
    sub.textContent = title;
    $("explorer").innerHTML = `<div class="status-box"><strong>${title}</strong>${detail}${r?.x ? ' <span class="bad">python-chess disagrees on legality.</span>' : ""}</div>`;
    $("best").disabled = true;
    return;
  }
  const moves = valuedMoves(r);
  const terminal = state.moves.length === 0;
  const mate = terminal && inCheck(state.board, state.turn);
  const whiteWins = (r.w === 2) === (state.turn === "w");
  pearl.className = `pearl ${r.w === 0 ? "" : whiteWins ? "win" : "loss"}`;
  if (r.w === 0) pearl.textContent = "½-½";
  else if (exact && r.r != null) pearl.textContent = mate ? "#" : `#${whiteWins ? "" : "-"}${Math.ceil(r.r / 2)}`;
  else pearl.textContent = whiteWins ? "1-0" : "0-1";
  gauge.classList.remove("is-off");
  $("gauge-black").style.height = r.w === 0 ? "50%" : whiteWins ? "0%" : "100%";
  const z = r.z === undefined ? "no Syzygy file" : Math.sign(r.z) === r.w / 2 ? "Syzygy agrees" : `Syzygy says ${SYZ[r.z]}`;
  const rankText = r.r == null ? "no proof rank" : exact ? `mate in ${r.r} plies` : `proof rank ${r.r}`;
  sub.textContent = mate ? `Checkmate · ${z}` : terminal ? `Stalemate · ${z}` : `${r.m} · ${rankText} · ${z} · ${r.ms} ms`;
  if (!moves) {
    $("explorer").innerHTML = '<div class="status-box"><strong>Move list mismatch</strong>The server and the client disagree on the legal moves; no values are shown.</div>';
    return;
  }
  renderExplorer(moves, exact);
  $("best").disabled = r.w === 0 || terminal;
}

function renderExplorer(moves, exact) {
  const box = $("explorer");
  box.replaceChildren();
  if (!moves.length) { box.innerHTML = '<p class="empty">No legal moves.</p>'; return; }
  const order = (a, b) => {
    const ra = a.rank ?? 1e9, rb = b.rank ?? 1e9;
    if (a.w === 2) return (b.mate - a.mate) || ra - rb;
    if (a.w === -2) return rb - ra;
    return 0;
  };
  for (const [title, w] of [["Winning", 2], ["Drawing", 0], ["Losing", -2], ["Unknown", null]]) {
    const list = moves.filter((m) => (m.w ?? null) === w).sort(order);
    if (!list.length) continue;
    const h = document.createElement("h3");
    h.textContent = `${title} · ${list.length}`;
    box.append(h);
    for (const m of list) box.append(moveRow(m, exact));
  }
  syncOverflow();
}

/** Table name of a board, colours swapped when only the mirrored table exists. */
function materialOf(board) {
  const side = (c) => `K${[..."QRBN"].map((r) => r.repeat(board.filter((p) => p === c + r).length)).join("")}`;
  const name = `${side("w")}v${side("b")}`;
  const swapped = `${side("b")}v${side("w")}`;
  return materials?.[name] ? name : swapped;
}

function moveRow(m, exact) {
  const b = document.createElement("button");
  b.className = "tb-move";
  const name = san(state.board, state.turn, m, state.moves);
  const label = m.mate ? "Checkmate" : m.stale ? "Stalemate" : RES[m.w] ?? "Unknown";
  // The child may live in a lower material: label its rank by that table.
  const childExact = materials?.[materialOf(m.next)]?.exactDtm ?? exact;
  const dist = m.rank != null && m.w !== 0 && !m.mate ? `${childExact ? "DTM" : "rank"} ${m.rank + 1}` : "";
  const warn = m.disagree ? '<span class="warn" title="Syzygy disagrees for this child">≠ Syzygy</span>' : "";
  b.innerHTML = `<span class="san">${name}${warn}</span><span class="dist">${dist}</span><span class="res ${m.mate ? "win" : cls(m.w)}">${label}</span>`;
  b.addEventListener("mouseenter", () => drawArrow(m));
  b.addEventListener("focus", () => drawArrow(m));
  b.addEventListener("mouseleave", clearArrows);
  b.addEventListener("blur", clearArrows);
  b.addEventListener("click", () => playMove(m));
  return b;
}

function syncOverflow() {
  const box = $("explorer");
  box.classList.toggle("has-more", box.scrollHeight - box.scrollTop - box.clientHeight > 4);
}
$("explorer").addEventListener("scroll", syncOverflow);
addEventListener("resize", syncOverflow);

function center(sq) {
  const f = sq % 8, r = sq >> 3;
  return state.flipped ? [7 - f + 0.5, r + 0.5] : [f + 0.5, 7 - r + 0.5];
}

function drawArrow(m) {
  clearArrows();
  const [x1, y1] = center(m.from), [x2, y2] = center(m.to);
  const len = Math.hypot(x2 - x1, y2 - y1), k = (len - 0.35) / len;
  const kind = m.mate || m.w === 2 ? "win" : m.w === -2 ? "loss" : "draw";
  const line = document.createElementNS(SVG, "line");
  for (const [a, v] of [["x1", x1], ["y1", y1], ["x2", x1 + (x2 - x1) * k], ["y2", y1 + (y2 - y1) * k]]) line.setAttribute(a, v);
  line.setAttribute("stroke", { win: "#15781b", draw: "#888", loss: "#882020" }[kind]);
  line.setAttribute("marker-end", `url(#head-${kind})`);
  $("arrows").append(line);
}

function clearArrows() {
  for (const l of $("arrows").querySelectorAll("line")) l.remove();
}

// ------------------------------------------------------------ move list (tview)
function renderTview() {
  const box = $("tview");
  box.replaceChildren();
  const start = parseFen(state.nodes[0]?.fen ?? START)?.turn ?? "w";
  let ply = start === "b" ? 1 : 0;
  for (let i = 1; i < state.nodes.length; i++, ply++) {
    if (ply % 2 === 0 || i === 1) {
      const idx = document.createElement("span");
      idx.className = "idx";
      idx.textContent = String((ply >> 1) + 1);
      box.append(idx);
      if (ply % 2 === 1) {
        const gap = document.createElement("span");
        gap.className = "gap";
        gap.textContent = "…";
        box.append(gap);
      }
    }
    const b = document.createElement("button");
    b.textContent = state.nodes[i].san;
    if (i === state.index) b.className = "active";
    b.addEventListener("click", () => go(i));
    box.append(b);
  }
  box.querySelector(".active")?.scrollIntoView({ block: "nearest" });
  $("first").disabled = $("prev").disabled = state.index === 0;
  $("next").disabled = $("last").disabled = state.index >= state.nodes.length - 1;
}

function go(i) {
  if (i < 0 || i >= state.nodes.length) return;
  state.index = i;
  show(state.nodes[i].fen);
}

async function bestLine() {
  $("best").disabled = true;
  const r = await fetch(`/api/line?fen=${encodeURIComponent(toFen(state.board, state.turn))}`).then((x) => x.json());
  if (r.s !== "ok" || !r.u.length) { $("best").disabled = false; return; }
  let board = state.board, turn = state.turn;
  state.nodes = state.nodes.slice(0, state.index + 1);
  for (const uci of r.u) {
    const all = legalMoves(board, turn);
    const m = all.find((x) => x.uci === uci);
    if (!m) break;
    state.nodes.push({ fen: toFen(m.next, other(turn)), uci, san: san(board, turn, m, all) });
    board = m.next;
    turn = other(turn);
  }
  go(state.nodes.length - 1);
}

// ------------------------------------------------------------ side column
function verifiedHtml(row) {
  if (row.verified === "pass") return `<span class="ok">${OK}pass</span>${row.runs > 1 ? ` <span class="muted">×${row.runs}</span>` : ""}`;
  if (row.verified === "fail") return `<span class="bad">${BAD}fail</span>`;
  return '<span class="pending">pending</span>';
}

function renderSide() {
  const r = state.result;
  const row = materials?.[r?.m];
  $("side-material").textContent = r?.m ?? "Analysis board";
  $("side-turn").textContent = `${SIDE[state.turn]} to play`;
  const facts = $("side-facts");
  facts.replaceChildren();
  if (!row) return;
  const add = (k, v) => {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k;
    dd.innerHTML = v;
    facts.append(dt, dd);
  };
  add("Positions", row.counts ? fmtInt(row.counts[0][0] + row.counts[1][0]) : "—");
  add("Solve", `${row.rounds} rounds · ${row.ranked ? "proof ranks" : "in place"}`);
  add("Verified", verifiedHtml(row));
  add("Generator", row.gen || "—");
  if (r.z !== undefined) add("Syzygy", Math.sign(r.z) === r.w / 2 ? `<span class="ok">${OK}agrees</span>` : `<span class="bad">${BAD}${SYZ[r.z]}</span>`);
}

// ------------------------------------------------------------ controls
function renderTurn() {
  for (const t of "wb") $(`turn-${t}`).setAttribute("aria-pressed", String(state.turn === t));
}
$("turn-w").addEventListener("click", () => { if (state.turn !== "w") { state.turn = "w"; edit(); } });
$("turn-b").addEventListener("click", () => { if (state.turn !== "b") { state.turn = "b"; edit(); } });
$("clear").addEventListener("click", () => { state.board.fill(null); edit(); });
$("flip").addEventListener("click", () => { state.flipped = !state.flipped; renderBoard(); });
$("first").addEventListener("click", () => go(0));
$("prev").addEventListener("click", () => go(state.index - 1));
$("next").addEventListener("click", () => go(state.index + 1));
$("last").addEventListener("click", () => go(state.nodes.length - 1));
$("best").addEventListener("click", bestLine);
$("fen-form").addEventListener("submit", (e) => { e.preventDefault(); show($("fen").value, { reset: true }); });
$("fen").addEventListener("change", () => show($("fen").value, { reset: true }));
$("random-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = e.submitter ?? $("random-form").querySelector("button");
  btn.disabled = true;
  const q = new URLSearchParams({ m: $("material").value, result: $("random-result").value });
  const r = await fetch(`/api/random?${q}`).then((x) => x.json());
  btn.disabled = false;
  if (r.s === "ok") show(r.f, { reset: true });
});

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  const keys = {
    ArrowLeft: () => go(state.index - 1),
    ArrowRight: () => go(state.index + 1),
    Home: () => go(0),
    End: () => go(state.nodes.length - 1),
    f: () => $("flip").click(),
    b: () => !$("best").disabled && bestLine(),
  };
  if (keys[e.key]) { keys[e.key](); e.preventDefault(); }
  if (e.key === "Escape") { state.tool = "move"; state.selected = null; syncTray(); renderBoard(); }
});

// ------------------------------------------------------------ views (loaded lazily)
for (const tab of document.querySelectorAll(".site-nav button")) tab.addEventListener("click", () => setView(tab.dataset.view));

function setView(view) {
  for (const tab of document.querySelectorAll(".site-nav button")) {
    tab.classList.toggle("is-active", tab.dataset.view === view);
    tab.setAttribute("aria-pressed", String(tab.dataset.view === view));
  }
  for (const v of ["analysis", "materials", "evidence"]) $(`view-${v}`).hidden = v !== view;
  const url = new URL(location.href);
  if (view === "analysis") url.searchParams.delete("view");
  else url.searchParams.set("view", view);
  history.replaceState(null, "", url);
  if (view === "materials") renderMaterials();
  if (view === "evidence") {
    if (state.result?.m && materials?.[state.result.m]) $("evidence-material").value = state.result.m;
    renderEvidence();
  }
}

const fmtInt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));
const fmtBytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(1)} kB`);

function shareBar([legal, w, d, l]) {
  const p = (x) => (legal ? (100 * x) / legal : 0);
  const seg = (v, c) => (v > 0 ? `<span style="flex:${v};background:${c}"></span>` : "");
  return `<span class="share" role="img" aria-label="win ${p(w).toFixed(1)}%, draw ${p(d).toFixed(1)}%, loss ${p(l).toFixed(1)}%">${seg(p(w), "#629924")}${seg(p(d), "#8a8a8a")}${seg(p(l), "#cc3333")}</span><span class="share-label">${p(w).toFixed(1)} / ${p(d).toFixed(1)} / ${p(l).toFixed(1)}</span>`;
}

function renderMaterials() {
  const t = $("materials-table");
  if (t.dataset.done || !materials) return;
  t.dataset.done = "1";
  const rows = Object.values(materials).map((m) => {
    const legal = m.counts ? m.counts[0][0] + m.counts[1][0] : null;
    return `<tr data-m="${m.name}"><td class="name">${m.name}</td><td>${fmtInt(legal)}</td><td>${m.counts ? shareBar(m.counts[0]) : "—"}</td><td>${m.counts ? shareBar(m.counts[1]) : "—"}</td><td>${m.rounds}</td><td>${m.ranked ? "ranked" : "in place"}</td><td>${verifiedHtml(m)}</td></tr>`;
  });
  t.innerHTML = `<thead><tr><th>Material</th><th>Legal positions</th><th>White to move · W / D / L %</th><th>Black to move · W / D / L %</th><th>Rounds</th><th>Mode</th><th>Verification</th></tr></thead><tbody>${rows.join("")}</tbody>`;
  t.querySelector("tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    $("material").value = tr.dataset.m;
    setView("analysis");
    $("random-form").requestSubmit();
  });
}

const evidenceCache = new Map();
async function renderEvidence() {
  const name = $("evidence-material").value;
  if (!name) return;
  if (!evidenceCache.has(name)) evidenceCache.set(name, fetch(`/api/evidence?m=${encodeURIComponent(name)}`).then((r) => r.json()));
  const ev = await evidenceCache.get(name);
  if ($("evidence-material").value !== name) return;
  const solve = ev.solve;
  drawCharts(solve);
  const kv = $("solve-facts");
  kv.replaceChildren();
  const rows = solve ? [
    ["Mode", solve.mode === "in-place-wdl" ? "in place (WDL only)" : "synchronous, proof ranks"],
    ["Rounds", `${Math.max(0, solve.changed.length - 2)} (+ seed and the zero-change round)`],
    ["Solve time", solve.seconds == null ? "—" : solve.seconds >= 1 ? `${solve.seconds.toFixed(1)} s` : `${(solve.seconds * 1000).toFixed(1)} ms`],
    ["Device memory", solve.bytes ? `${fmtBytes(solve.bytes)} allocated` : "—"],
    ["Payload SHA-256", `<code>${ev.digest}</code>`],
    ["Generator", ev.generator || "—"],
    ["Log", solve.log],
  ] : [["Solve", "no solver log for this table"]];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k;
    dd.innerHTML = v;
    kv.append(dt, dd);
  }
  $("verify-list").innerHTML = ev.verify?.length ? ev.verify.map((c) => `
    <div class="box report">
      <header><strong class="${c.status === "pass" ? "ok" : "bad"}">${c.status === "pass" ? OK : BAD}${c.status}</strong><span>${c.scope?.startsWith("canonical") ? "canonical positions + 8 symmetries" : "all positions"}</span></header>
      <p>${fmtInt(c.legal_positions)} legal positions; ${fmtInt(c.syzygy_wdl_compared)} compared with Syzygy</p>
      <p>${c.proof_ranks_present === false ? "No proof ranks (in-place table)" : `${fmtInt(c.proof_obligations_checked)} proof obligations checked`}${c.symmetry_violating_words != null ? `; ${fmtInt(c.symmetry_violating_words)} symmetry violations` : ""}</p>
      ${c.first_failure ? `<p class="bad">First failure: ${c.first_failure}</p>` : ""}
      <p class="src">${c.log} · ${c.seconds?.toFixed(1)} s</p>
    </div>`).join("") : '<div class="box report"><span class="pending">Pending.</span> No verifier report matches this payload digest yet.</div>';
  const cards = [];
  if (ev.dtm) cards.push(`<div class="box report"><header><strong>This table vs lichess DTM</strong><span>${ev.dtm.a} / ${ev.dtm.n} agree</span></header><p>WDL and |DTM| plies vs the synchronous proof rank; matched by payload digest.</p></div>`);
  if (ev.ref) {
    const extra = [ev.ref.u ? `${ev.ref.u} unresolved (no local reference value), excluded` : "", ev.ref.s ? `${ev.ref.s} with a nonzero halfmove clock, recorded separately` : ""].filter(Boolean).join("; ");
    cards.push(`<div class="box report"><header><strong>Syzygy reference vs lichess</strong><span class="${ev.ref.d ? "bad" : ""}">${ev.ref.a} / ${ev.ref.a + ev.ref.d} compared agree</span></header><p>Checks the reference files this table is verified against, not the table itself.</p>${extra ? `<p>${extra}</p>` : ""}</div>`);
  }
  $("online-list").innerHTML = cards.join("") || `<div class="box report">No online comparison covers ${name}.</div>`;
}

function svgEl(tag, attrs, text) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
}
const shortNum = (v) => (v >= 1e9 ? `${v / 1e9}G` : v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : String(v));
const fmtT = (ms) => (ms === 0 ? "0" : ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s` : ms >= 10 ? `${ms.toFixed(0)} ms` : `${Number(ms.toPrecision(2))} ms`);

function drawCharts(solve) {
  for (const id of ["chart-changed", "chart-time"]) $(id).replaceChildren();
  if (!solve?.changed?.length) { $("chart-changed").innerHTML = '<p class="empty">No per-round record.</p>'; return; }
  const n = solve.changed.length;
  const W = Math.max(320, $("chart-changed").clientWidth), H = 240, P = { l: 52, r: 16, t: 12, b: 26 }, pad = 14;
  const x = (i) => P.l + pad + (n === 1 ? 0 : (i * (W - P.l - P.r - 2 * pad)) / (n - 1));
  const step = [1, 2, 5, 10, 20, 50].find((s) => (n - 1) / s <= 12) ?? 100;
  const ticks = [];
  for (let i = 0; i < n; i += step) ticks.push(i);
  const top = Math.ceil(Math.log10(Math.max(...solve.changed, 1)));
  const y = (v) => H - P.b - ((v <= 0 ? 0 : Math.log10(v) + 1) * (H - P.t - P.b)) / (top + 1);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Words changed per sweep, log scale" });
  for (let p = 0; p <= top; p += top > 6 ? 2 : 1) {
    svg.append(svgEl("line", { class: "grid", x1: P.l, x2: W - P.r, y1: y(10 ** p), y2: y(10 ** p) }));
    svg.append(svgEl("text", { class: "axis", x: P.l - 8, y: y(10 ** p) + 4, "text-anchor": "end" }, shortNum(10 ** p)));
  }
  svg.append(svgEl("text", { class: "axis", x: P.l - 8, y: y(0) + 4, "text-anchor": "end" }, "0"));
  for (const i of ticks) svg.append(svgEl("text", { class: "axis", x: x(i), y: H - 6, "text-anchor": "middle" }, String(i)));
  svg.append(svgEl("path", { class: "series", d: solve.changed.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("") }));
  solve.changed.forEach((v, i) => { if (v === 0) svg.append(svgEl("circle", { class: "zero", cx: x(i), cy: y(0), r: 5 })); });
  const cross = svgEl("line", { class: "cross", y1: P.t, y2: H - P.b, visibility: "hidden" });
  const dot = svgEl("circle", { class: "dot", r: 5, visibility: "hidden" });
  const hit = svgEl("rect", { class: "hit", x: P.l - 10, y: 0, width: W - P.l - P.r + 20, height: H });
  svg.append(cross, dot, hit);
  $("chart-changed").append(svg);

  const times = solve.sweep_ms;
  const raw = Math.max(...times, 1e-6), mag = 10 ** Math.floor(Math.log10(raw));
  const maxT = [1, 2, 5, 10].map((k) => k * mag).find((v) => v >= raw);
  const H2 = 120, y2 = (v) => H2 - P.b - (v * (H2 - P.t - P.b)) / maxT;
  const svg2 = svgEl("svg", { viewBox: `0 0 ${W} ${H2}`, role: "img", "aria-label": "Sweep time per round" });
  for (const f of [0, 0.5, 1]) {
    svg2.append(svgEl("line", { class: "grid", x1: P.l, x2: W - P.r, y1: y2(maxT * f), y2: y2(maxT * f) }));
    svg2.append(svgEl("text", { class: "axis", x: P.l - 8, y: y2(maxT * f) + 4, "text-anchor": "end" }, fmtT(maxT * f)));
  }
  const bw = Math.max(2, Math.min(18, (W - P.l - P.r - 2 * pad) / n - 2));
  times.forEach((v, i) => {
    const h = Math.max(1, H2 - P.b - y2(v));
    svg2.append(svgEl("rect", { class: "bar", x: x(i) - bw / 2, y: H2 - P.b - h, width: bw, height: h, rx: Math.min(2, bw / 2) }));
  });
  for (const i of ticks) svg2.append(svgEl("text", { class: "axis", x: x(i), y: H2 - 6, "text-anchor": "middle" }, String(i)));
  const hit2 = svgEl("rect", { class: "hit", x: P.l - 10, y: 0, width: W - P.l - P.r + 20, height: H2 });
  svg2.append(hit2);
  $("chart-time").append(svg2);

  const tip = $("tooltip");
  const move = (e, node) => {
    const box = node.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - P.l - pad) / (W - P.l - P.r - 2 * pad)) * (n - 1))));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(i));
    dot.setAttribute("cy", y(solve.changed[i]));
    dot.setAttribute("visibility", "visible");
    tip.innerHTML = `Round ${i}${i === 0 ? " (checkmate seed)" : ""}<br>${fmtInt(solve.changed[i])} words changed<br>sweep ${fmtT(times[i])}${i < 2 ? " (includes JIT)" : ""}`;
    tip.hidden = false;
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - 220)}px`;
    tip.style.top = `${e.clientY + 14}px`;
  };
  const leave = () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dot.setAttribute("visibility", "hidden");
  };
  hit.addEventListener("pointermove", (e) => move(e, svg));
  hit2.addEventListener("pointermove", (e) => move(e, svg2));
  hit.addEventListener("pointerleave", leave);
  hit2.addEventListener("pointerleave", leave);
}

$("evidence-material").addEventListener("change", renderEvidence);

// ------------------------------------------------------------ boot
async function boot() {
  renderTray();
  const fromHash = decodeURIComponent(location.hash.slice(1));
  const first = show(parseFen(fromHash) ? fromHash : START, { reset: true });
  const list = await fetch("/api/materials").then((r) => r.json());
  materials = Object.fromEntries(list.rows.map((row) => {
    const o = Object.fromEntries(list.cols.map((c, i) => [c, row[i]]));
    return [o.name, o];
  }));
  const names = Object.keys(materials);
  $("material").innerHTML = names.map((n) => `<option>${n}</option>`).join("");
  $("evidence-material").innerHTML = $("material").innerHTML;
  if (names.length) $("material").value = $("evidence-material").value = names.at(-1);
  await first;
  renderTools();
  renderSide();
  const view = new URLSearchParams(location.search).get("view");
  if (view === "materials" || view === "evidence") setView(view);
}
boot();
