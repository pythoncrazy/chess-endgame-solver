// Tablebase visualizer: lichess-style analysis over the memory-mapped GPU tables.
// The server returns every legal move with its child value, so the client keeps no
// chess rules of its own beyond FEN placement.

const $ = (id) => document.getElementById(id);
const FILES = "abcdefgh";
const SIDE = { w: "White", b: "Black" };
const START = "8/8/8/8/4k3/8/8/KQ6 w - - 0 1";

const state = {
  board: new Array(64).fill(null), // "wK", "bR", ... indexed a1 = 0
  turn: "w",
  flipped: false,
  tool: "move", // "move" | "trash" | piece code to place
  selected: null,
  result: null,
  history: [], // [{ fen, san, uci }]; index 0 is the setup position
  index: 0,
  lastMove: null,
  request: 0,
};
const cache = new Map();

// ---------------------------------------------------------------- FEN helpers
function parseFen(fen) {
  const [placement, turn] = fen.trim().split(/\s+/);
  const rows = placement?.split("/");
  if (!rows || rows.length !== 8 || !["w", "b"].includes(turn)) return null;
  const board = new Array(64).fill(null);
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (/[1-8]/.test(ch)) { file += Number(ch); continue; }
      if (!/[kqrbnpKQRBNP]/.test(ch) || file > 7) return null;
      const color = ch === ch.toUpperCase() ? "w" : "b";
      board[(7 - r) * 8 + file] = color + ch.toUpperCase();
      file += 1;
    }
    if (file !== 8) return null;
  }
  return { board, turn };
}

function buildFen(board, turn) {
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

const sqName = (sq) => FILES[sq % 8] + (Math.floor(sq / 8) + 1);
const sqIndex = (name) => FILES.indexOf(name[0]) + 8 * (Number(name[1]) - 1);

// ---------------------------------------------------------------- probing
function probe(fen) {
  if (!cache.has(fen)) {
    const p = fetch(`/api/probe?fen=${encodeURIComponent(fen)}`)
      .then((r) => r.json())
      .catch(() => ({ status: "error", message: "The server did not answer. Is it still running?" }));
    cache.set(fen, p);
    if (cache.size > 4000) cache.delete(cache.keys().next().value);
  }
  return cache.get(fen);
}

async function show(fen, { push = null, replace = false } = {}) {
  const parsed = parseFen(fen);
  if (!parsed) { $("fen").classList.add("is-bad"); return; }
  $("fen").classList.remove("is-bad");
  state.board = parsed.board;
  state.turn = parsed.turn;
  state.selected = null;
  if (push) {
    state.history = state.history.slice(0, state.index + 1);
    state.history.push({ fen, ...push });
    state.index = state.history.length - 1;
  } else if (replace) {
    state.history = [{ fen, san: null, uci: null }];
    state.index = 0;
  }
  const entry = state.history[state.index];
  state.lastMove = entry?.uci ? [sqIndex(entry.uci.slice(0, 2)), sqIndex(entry.uci.slice(2, 4))] : null;
  history.replaceState(null, "", `#${encodeURIComponent(fen)}`);
  $("fen").value = fen;
  renderTurn();
  renderBoard();
  renderLine();
  const id = ++state.request;
  const slow = setTimeout(() => id === state.request && renderLoading(), 120);
  const result = await probe(fen);
  clearTimeout(slow);
  if (id !== state.request) return;
  state.result = result;
  renderBoard();
  renderPanel();
}

const edit = () => show(buildFen(state.board, state.turn), { replace: true });

// ---------------------------------------------------------------- board
function renderBoard() {
  const el = $("board");
  const res = state.result?.fen === buildFen(state.board, state.turn) ? state.result : null;
  const dests = new Set();
  if (state.selected !== null && res?.moves) {
    for (const m of res.moves) if (sqIndex(m.uci.slice(0, 2)) === state.selected) dests.add(sqIndex(m.uci.slice(2, 4)));
  }
  const checkSq = res?.check ? state.board.findIndex((p) => p === `${state.turn}K`) : -1;
  const frag = document.createDocumentFragment();
  for (let i = 0; i < 64; i++) {
    const row = Math.floor(i / 8), col = i % 8;
    const sq = state.flipped ? row * 8 + (7 - col) : (7 - row) * 8 + col;
    const cell = document.createElement("div");
    const light = (Math.floor(sq / 8) + (sq % 8)) % 2 === 1;
    cell.className = `sq ${light ? "light" : "dark"}`;
    cell.dataset.sq = sq;
    cell.setAttribute("role", "gridcell");
    const piece = state.board[sq];
    cell.setAttribute("aria-label", `${sqName(sq)}${piece ? ` ${SIDE[piece[0]]} ${pieceName(piece[1])}` : ""}`);
    if (state.lastMove?.includes(sq)) cell.classList.add("last");
    if (state.selected === sq) cell.classList.add("sel");
    if (dests.has(sq)) cell.classList.add("dest");
    if (dests.has(sq) && piece) cell.classList.add("occupied");
    if (sq === checkSq) cell.classList.add("check");
    if (piece) {
      const img = document.createElement("img");
      img.src = `/piece/${piece}.svg`;
      img.alt = "";
      img.draggable = false;
      cell.append(img);
    }
    if (col === 7) cell.append(coord("rank", String(Math.floor(sq / 8) + 1)));
    if (row === 7) cell.append(coord("file", FILES[sq % 8]));
    frag.append(cell);
  }
  el.replaceChildren(frag);
}

function coord(kind, text) {
  const s = document.createElement("span");
  s.className = `coord ${kind}`;
  s.textContent = text;
  return s;
}

const pieceName = (r) => ({ K: "king", Q: "queen", R: "rook", B: "bishop", N: "knight", P: "pawn" })[r];

function squareAt(x, y) {
  const cell = document.elementFromPoint(x, y)?.closest(".sq");
  return cell && $("board").contains(cell) ? Number(cell.dataset.sq) : null;
}

function legalMove(from, to) {
  const res = state.result;
  if (res?.status !== "ok") return null;
  return res.moves.find((m) => m.uci === sqName(from) + sqName(to)) ?? null;
}

function playMove(m) {
  state.selected = null;
  show(m.fen, { push: { san: m.san, uci: m.uci } });
}

let drag = null;
$("board").addEventListener("pointerdown", (e) => {
  if (e.button !== 0) return;
  const sq = squareAt(e.clientX, e.clientY);
  if (sq === null) return;
  const piece = state.board[sq];
  if (state.tool === "trash") { if (piece) { state.board[sq] = null; edit(); } return; }
  if (state.tool !== "move") {
    state.board[sq] = piece === state.tool ? null : state.tool;
    edit();
    return;
  }
  if (state.selected !== null && state.selected !== sq) {
    const m = legalMove(state.selected, sq);
    if (m) { playMove(m); return; }
  }
  if (!piece) { state.selected = null; renderBoard(); return; }
  state.selected = piece[0] === state.turn ? sq : null;
  renderBoard();
  const img = document.createElement("img");
  img.src = `/piece/${piece}.svg`;
  img.className = "drag-piece";
  const size = $("board").clientWidth / 8;
  img.style.width = img.style.height = `${size}px`;
  drag = { from: sq, piece, img, size, moved: false, x: e.clientX, y: e.clientY };
  $("board").setPointerCapture(e.pointerId);
});

$("board").addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (!drag.moved && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 4) return;
  if (!drag.moved) {
    drag.moved = true;
    document.body.append(drag.img);
    $("board").querySelector(`[data-sq="${drag.from}"] img`)?.classList.add("ghost");
  }
  drag.img.style.left = `${e.clientX - drag.size / 2}px`;
  drag.img.style.top = `${e.clientY - drag.size / 2}px`;
});

$("board").addEventListener("pointerup", (e) => {
  if (!drag) return;
  const d = drag;
  drag = null;
  d.img.remove();
  if (!d.moved) return;
  const to = squareAt(e.clientX, e.clientY);
  if (to === d.from) { renderBoard(); return; }
  const m = to === null ? null : legalMove(d.from, to);
  if (m) { playMove(m); return; }
  // Not a legal move: treat the drop as an edit; off the board removes the piece.
  state.board[d.from] = null;
  if (to !== null) state.board[to] = d.piece;
  edit();
});

$("board").addEventListener("contextmenu", (e) => {
  e.preventDefault();
  const sq = squareAt(e.clientX, e.clientY);
  if (sq !== null && state.board[sq]) { state.board[sq] = null; edit(); }
});

// ---------------------------------------------------------------- tray
function renderTray() {
  for (const row of document.querySelectorAll(".tray-row")) {
    const color = row.dataset.color;
    row.replaceChildren();
    for (const r of ["K", "Q", "R", "B", "N"]) {
      const code = color + r;
      const b = document.createElement("button");
      b.className = "tray-btn";
      b.dataset.tool = code;
      b.title = `Place ${SIDE[color].toLowerCase()} ${pieceName(r)}`;
      b.setAttribute("aria-label", b.title);
      b.innerHTML = `<img src="/piece/${code}.svg" alt="">`;
      row.append(b);
    }
    if (color === "w") {
      row.append(toolButton("move", "Move pieces (drag off the board to remove)",
        '<path d="M5 3l14 8-6 2-2 6z"/>'));
      row.append(toolButton("trash", "Remove pieces",
        '<path d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13"/>'));
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

// ---------------------------------------------------------------- panel
const VERDICT = { 2: "wins", 0: "draw", "-2": "loses" };
const cls = (wdl) => (wdl === 2 ? "win" : wdl === -2 ? "loss" : wdl === 0 ? "draw" : "unknown");
const SYZ = { 2: "Win", 1: "Cursed win", 0: "Draw", "-1": "Blessed loss", "-2": "Loss" };
const ICON_OK = '<svg class="mark" viewBox="0 0 16 16" aria-hidden="true"><path d="m3 8.5 3 3 7-7"/></svg>';
const ICON_BAD = '<svg class="mark" viewBox="0 0 16 16" aria-hidden="true"><path d="m4 4 8 8m0-8-8 8"/></svg>';

function renderLoading() {
  $("moves").innerHTML = '<div class="skeleton"></div>'.repeat(6);
}

function renderPanel() {
  const r = state.result;
  const verdict = $("verdict"), sub = $("verdict-sub"), facts = $("facts");
  facts.replaceChildren();
  clearArrows();
  if (!r || r.status !== "ok") {
    verdict.className = "verdict warn";
    const titles = {
      illegal: "Illegal position",
      unsupported: "Outside the tables",
      "missing-table": "No table for this material",
      "invalid-fen": "Invalid FEN",
      error: "Server unavailable",
    };
    verdict.textContent = titles[r?.status] ?? "No result";
    sub.textContent = r?.message ?? "";
    if (r?.table) addFacts([["Table", tableFact(r.table)]]);
    if (r?.agrees_with_python_chess === false) addFacts([["python-chess", `<span class="bad">${ICON_BAD}disagrees on legality</span>`]]);
    $("moves").innerHTML = `<p class="empty">${emptyHint(r)}</p>`;
    $("best-line").disabled = true;
    return;
  }
  const side = SIDE[r.turn], other = SIDE[r.turn === "w" ? "b" : "w"];
  verdict.className = `verdict ${r.checkmate ? "loss" : cls(r.wdl)}`;
  if (r.checkmate) {
    verdict.textContent = `Checkmate · ${other} wins`;
  } else if (r.stalemate) {
    verdict.textContent = "Stalemate · draw";
  } else {
    verdict.textContent = r.wdl === 0 ? "Draw" : `${side} ${VERDICT[r.wdl]}`;
  }
  sub.textContent = rankText(r);
  addFacts([
    ["Syzygy", syzygyFact(r.syzygy, r.wdl)],
    ["Table", tableFact(r.table)],
    ["Verified", verifiedFact(r.table)],
    ["Probe", `${r.probe_ms} ms on the server, including the Syzygy check of every move`],
  ]);
  renderMoves(r);
  $("best-line").disabled = !r.moves.length || r.wdl === 0;
}

function emptyHint(r) {
  if (r?.status === "illegal") return "Move the king out of check, separate the kings, or switch the side to move.";
  if (r?.status === "unsupported") return "The tables cover pawnless positions without castling rights, two kings and up to four other pieces.";
  if (r?.status === "missing-table") return "Solve this material with gpu_solve, then restart the server with its output directory.";
  return "Place pieces from the tray, paste a FEN, or load a random position.";
}

function rankText(r) {
  if (r.wdl === 0) return r.stalemate ? "No legal move and not in check." : "Neither side can force mate. Unclocked: the fifty-move rule is not applied.";
  if (r.rank == null) return "Unclocked WDL. Six-piece tables are solved in place and carry no proof ranks.";
  const moves = Math.ceil(r.rank / 2);
  if (r.table.exact_dtm) {
    if (r.rank === 0) return "Mate on the board.";
    return r.wdl === 2 ? `Mate in ${moves} (${r.rank} plies) with best play.` : `Mated in ${moves} (${r.rank} plies) with best defence.`;
  }
  return `Proof rank ${r.rank}: decided in ${r.rank} synchronous rounds; captures count as rank 0, so this is not distance to mate.`;
}

function syzygyFact(s, wdl) {
  if (!s) return '<span>no reference file for this material</span>';
  const text = `${SYZ[s.wdl]} (Syzygy WDL ${s.wdl > 0 ? "+" : ""}${s.wdl})`;
  return s.agrees
    ? `<span class="ok">${ICON_OK}agrees</span> · ${text}`
    : `<span class="bad">${ICON_BAD}disagrees</span> · ${text}, table says ${SYZ[wdl]}`;
}

function tableFact(t) {
  const gen = t.generator ? ` · ${t.generator.replace(/git:([0-9a-f]{8})[0-9a-f]+/, "git:$1")}` : "";
  return `${t.material}${gen}`;
}

function verifiedFact(t) {
  if (!t.verified?.length) return '<span class="pending">pending · no verifier report for this payload digest yet</span>';
  return t.verified.map((v) => {
    const ok = v.status === "pass";
    const scope = v.scope?.startsWith("canonical") ? "canonical + 8 symmetries" : "all positions";
    return `<span class="${ok ? "ok" : "bad"}">${ok ? ICON_OK : ICON_BAD}${v.status}</span> · ${scope} · ${fmtInt(v.legal_positions)} legal checked`;
  }).join("<br>");
}

function addFacts(rows) {
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k;
    dd.innerHTML = v;
    $("facts").append(dt, dd);
  }
}

function renderMoves(r) {
  const groups = [
    ["Winning moves", (m) => m.wdl === 2],
    ["Drawing moves", (m) => m.wdl === 0],
    ["Losing moves", (m) => m.wdl === -2],
    ["Unknown", (m) => m.wdl == null],
  ];
  const box = $("moves");
  box.replaceChildren();
  if (!r.moves.length) {
    box.innerHTML = `<p class="empty">${r.checkmate ? "Checkmate: no legal moves." : "Stalemate: no legal moves."}</p>`;
    return;
  }
  for (const [title, test] of groups) {
    const list = r.moves.filter(test);
    if (!list.length) continue;
    const g = document.createElement("div");
    g.className = "move-group";
    g.innerHTML = `<h2>${title}</h2>`;
    for (const m of list) g.append(moveRow(m));
    box.append(g);
  }
}

function moveRow(m) {
  const b = document.createElement("button");
  b.className = "move";
  const label = m.checkmate ? "Checkmate" : m.stalemate ? "Stalemate" : { 2: "Win", 0: "Draw", "-2": "Loss" }[m.wdl] ?? "Unknown";
  const plies = m.child_rank != null && m.wdl !== 0 ? `${m.child_rank + 1} plies` : m.capture ? "capture" : "";
  const warn = m.syzygy_agrees === false ? ' <span class="warn" title="Syzygy disagrees with the table for this child">≠ Syzygy</span>' : "";
  b.innerHTML = `<span class="san">${m.san}${warn}</span><span class="rank">${plies}</span><span class="badge ${m.checkmate ? "win" : cls(m.wdl)}">${label}</span>`;
  b.title = m.child_table && m.child_table !== state.result.material ? `Child probed in ${m.child_table}` : "";
  b.addEventListener("mouseenter", () => { drawArrow(m); probe(m.fen); });
  b.addEventListener("focus", () => { drawArrow(m); probe(m.fen); });
  b.addEventListener("mouseleave", clearArrows);
  b.addEventListener("blur", clearArrows);
  b.addEventListener("click", () => playMove(m));
  return b;
}

function center(sq) {
  const f = sq % 8, r = Math.floor(sq / 8);
  return state.flipped ? [7 - f + 0.5, r + 0.5] : [f + 0.5, 7 - r + 0.5];
}

function drawArrow(m) {
  clearArrows();
  const [x1, y1] = center(sqIndex(m.uci.slice(0, 2)));
  const [x2, y2] = center(sqIndex(m.uci.slice(2, 4)));
  const len = Math.hypot(x2 - x1, y2 - y1);
  const k = (len - 0.35) / len;
  const kind = m.checkmate ? "win" : cls(m.wdl) === "unknown" ? "draw" : cls(m.wdl);
  const color = { win: "#629924", draw: "#9a9a9a", loss: "#cc3333" }[kind];
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1); line.setAttribute("y1", y1);
  line.setAttribute("x2", x1 + (x2 - x1) * k); line.setAttribute("y2", y1 + (y2 - y1) * k);
  line.setAttribute("stroke", color);
  line.setAttribute("marker-end", `url(#head-${kind})`);
  $("arrows").append(line);
}

function clearArrows() {
  for (const l of $("arrows").querySelectorAll("line")) l.remove();
}

// ---------------------------------------------------------------- line / history
function renderLine() {
  const ol = $("line");
  ol.replaceChildren();
  const startTurn = state.history[0] ? parseFen(state.history[0].fen)?.turn : "w";
  state.history.forEach((h, i) => {
    if (i === 0) return;
    const ply = i - 1 + (startTurn === "b" ? 1 : 0);
    if (ply % 2 === 0 || i === 1) {
      const n = document.createElement("li");
      n.className = "num";
      n.textContent = `${Math.floor(ply / 2) + 1}.${ply % 2 ? ".." : ""}`;
      ol.append(n);
    }
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.textContent = h.san;
    if (i === state.index) b.className = "is-current";
    b.addEventListener("click", () => go(i));
    li.append(b);
    ol.append(li);
  });
  $("back").disabled = state.index === 0;
  $("forward").disabled = state.index >= state.history.length - 1;
  ol.querySelector(".is-current")?.scrollIntoView({ block: "nearest" });
}

function go(i) {
  if (i < 0 || i >= state.history.length) return;
  state.index = i;
  show(state.history[i].fen);
}

async function bestLine() {
  const btn = $("best-line");
  btn.disabled = true;
  for (let ply = 0; ply < 80; ply++) {
    const r = await probe(buildFen(state.board, state.turn));
    if (r.status !== "ok" || !r.moves.length || r.wdl === 0) break;
    const m = r.moves[0];
    await show(m.fen, { push: { san: m.san, uci: m.uci } });
  }
  btn.disabled = false;
}

// ---------------------------------------------------------------- controls
function renderTurn() {
  for (const t of ["w", "b"]) {
    const b = $(`turn-${t}`);
    b.classList.toggle("is-on", state.turn === t);
    b.setAttribute("aria-pressed", String(state.turn === t));
  }
}

$("turn-w").addEventListener("click", () => { if (state.turn !== "w") { state.turn = "w"; edit(); } });
$("turn-b").addEventListener("click", () => { if (state.turn !== "b") { state.turn = "b"; edit(); } });
$("flip").addEventListener("click", () => { state.flipped = !state.flipped; renderBoard(); clearArrows(); });
$("clear").addEventListener("click", () => { state.board.fill(null); edit(); });
$("back").addEventListener("click", () => go(state.index - 1));
$("forward").addEventListener("click", () => go(state.index + 1));
$("best-line").addEventListener("click", bestLine);
$("fen-form").addEventListener("submit", (e) => { e.preventDefault(); show($("fen").value, { replace: true }); });
$("fen").addEventListener("change", () => show($("fen").value, { replace: true }));
$("copy-fen").addEventListener("click", () => navigator.clipboard?.writeText($("fen").value));
$("random-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = e.submitter;
  btn.disabled = true;
  const q = new URLSearchParams({ material: $("material").value, result: $("random-result").value, turn: "any" });
  const r = await fetch(`/api/random?${q}`).then((x) => x.json());
  btn.disabled = false;
  if (r.status === "ok") show(r.fen, { replace: true });
  else { $("verdict").className = "verdict warn"; $("verdict").textContent = "No position found"; $("verdict-sub").textContent = r.message; }
});

document.addEventListener("keydown", (e) => {
  if (e.target.closest("input, select, textarea")) return;
  if (e.key === "ArrowLeft") { go(state.index - 1); e.preventDefault(); }
  if (e.key === "ArrowRight") { go(state.index + 1); e.preventDefault(); }
  if (e.key === "f") $("flip").click();
  if (e.key === "Escape") { state.tool = "move"; state.selected = null; syncTray(); renderBoard(); }
});

// ---------------------------------------------------------------- views
for (const tab of document.querySelectorAll(".nav-tab")) {
  tab.addEventListener("click", () => setView(tab.dataset.view));
}

function setView(view) {
  for (const tab of document.querySelectorAll(".nav-tab")) {
    const on = tab.dataset.view === view;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-pressed", String(on));
  }
  for (const v of ["analysis", "materials", "evidence"]) $(`view-${v}`).hidden = v !== view;
  const url = new URL(location.href);
  if (view === "analysis") url.searchParams.delete("view"); else url.searchParams.set("view", view);
  history.replaceState(null, "", url);
  if (view === "evidence") {
    const current = state.result?.material;
    if (current && evidence?.solves[current]) $("evidence-material").value = current;
    renderEvidence();
  }
}

// ---------------------------------------------------------------- materials
let materials = [];
let evidence = null;
const fmtBytes = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(1)} kB`);
const fmtInt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));
const pct = (x, total) => (total ? (100 * x) / total : 0);

function shareBar(c) {
  const total = c.legal || 0;
  const w = pct(c.win, total), d = pct(c.draw, total), l = pct(c.loss, total);
  const seg = (v, color) => (v > 0 ? `<span style="flex:${v};background:${color}"></span>` : "");
  return `<div class="share-cell"><div class="share" role="img" aria-label="win ${w.toFixed(1)}%, draw ${d.toFixed(1)}%, loss ${l.toFixed(1)}%">${seg(w, "#629924")}${seg(d, "#8a8a8a")}${seg(l, "#cc3333")}</div><span class="share-label">${w.toFixed(1)} / ${d.toFixed(1)} / ${l.toFixed(1)}</span></div>`;
}

function verificationSummary(list) {
  if (!list.length) return '<span class="pending">pending</span>';
  const failed = list.filter((x) => x.status !== "pass").length;
  const runs = list.length > 1 ? ` <span class="muted">×${list.length}</span>` : "";
  return failed
    ? `<span class="status bad">${ICON_BAD}fail</span>${runs}`
    : `<span class="status ok">${ICON_OK}pass</span>${runs}`;
}

function renderMaterials() {
  const t = $("materials-table");
  const head = "<thead><tr><th>Material</th><th>Legal positions</th><th>White to move · W / D / L %</th><th>Black to move · W / D / L %</th><th>Rounds</th><th>Mode</th><th>Verification</th></tr></thead>";
  const rows = materials.map((m) => {
    const c = m.solve?.counts;
    const legal = c ? c[0].legal + c[1].legal : null;
    const v = verificationSummary(m.verified);
    return `<tr data-material="${m.material}"><td class="name">${m.material}</td><td>${fmtInt(legal)}</td><td>${c ? shareBar(c[0]) : "—"}</td><td>${c ? shareBar(c[1]) : "—"}</td><td>${m.rounds}</td><td>${m.ranked ? "ranked" : "in place"}</td><td>${v}</td></tr>`;
  });
  t.innerHTML = `${head}<tbody>${rows.join("")}</tbody>`;
  t.querySelector("tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr) return;
    $("material").value = tr.dataset.material;
    setView("analysis");
    $("random-form").requestSubmit();
  });
}

// ---------------------------------------------------------------- evidence
function renderEvidence() {
  if (!evidence) return;
  const name = $("evidence-material").value;
  const solve = evidence.solves[name];
  const info = materials.find((m) => m.material === name);
  drawCharts(solve);
  const kv = $("solve-facts");
  kv.replaceChildren();
  const rows = solve ? [
    ["Mode", solve.mode === "in-place-wdl" ? "in place (WDL only)" : "synchronous, proof ranks"],
    ["Rounds", `${solve.rounds} (+ seed and the zero-change round)`],
    ["Solve time", solve.rounds_seconds == null ? "—" : solve.rounds_seconds >= 1 ? `${solve.rounds_seconds.toFixed(1)} s` : `${(solve.rounds_seconds * 1000).toFixed(1)} ms`],
    ["Device memory", solve.device_bytes ? `${fmtBytes(solve.device_bytes)} allocated` : "—"],
    ["Payload SHA-256", `<code>${solve.digest}</code>`],
    ["Generator", info?.generator || "—"],
    ["Log", solve.log],
  ] : [["Solve", "no solver log for this material"]];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = k; dd.innerHTML = v;
    kv.append(dt, dd);
  }
  const checks = info ? evidence.verifications[info.digest] ?? [] : [];
  $("verify-list").innerHTML = checks.length ? `<div class="card-list">${checks.map((c) => `
    <div class="report">
      <div class="report-head"><strong class="${c.status === "pass" ? "ok" : "bad"}">${c.status === "pass" ? ICON_OK : ICON_BAD}${c.status}</strong><span>${c.scope?.startsWith("canonical") ? "canonical + symmetry" : "all positions"}</span></div>
      <p>${fmtInt(c.legal_positions)} legal positions; ${fmtInt(c.syzygy_wdl_compared)} compared with Syzygy</p>
      <p>${c.proof_ranks_present === false ? "No proof ranks (in-place table)" : `${fmtInt(c.proof_obligations_checked)} proof obligations checked`}${c.symmetry_violating_words != null ? `; ${fmtInt(c.symmetry_violating_words)} symmetry violations` : ""}</p>
      ${c.first_failure ? `<p class="bad">First failure: ${c.first_failure}</p>` : ""}
      <p class="src">${c.log} · ${c.seconds?.toFixed(1)} s</p>
    </div>`).join("")}</div>` : '<div class="report empty-report"><span class="pending">Pending.</span> No verifier report matches this table\'s payload digest yet. Run verify_pawnless on the table file; results appear here by digest.</div>';
  const cards = [];
  const dtm = info ? evidence.dtm_checks[info.digest] : null;
  if (dtm) {
    cards.push(`<div class="report"><div class="report-head"><strong>This table vs lichess DTM</strong><span>${fmtInt(dtm.agree)} / ${fmtInt(dtm.compared)} agree</span></div><p>${dtm.metric}; matched by payload digest.</p></div>`);
  }
  const ref = evidence.reference_checks[name];
  if (ref) {
    const compared = ref.agree + ref.differ;
    const extra = [
      ref.unresolved ? `${ref.unresolved} unresolved (no local reference value), excluded` : "",
      ref.recorded_separately ? `${ref.recorded_separately} with a nonzero halfmove clock, recorded separately` : "",
    ].filter(Boolean).join("; ");
    cards.push(`<div class="report"><div class="report-head"><strong>Syzygy reference vs lichess</strong><span class="${ref.differ ? "bad" : ""}">${fmtInt(ref.agree)} / ${fmtInt(compared)} compared agree</span></div><p>Checks the reference files this table is verified against, not the table itself.</p>${extra ? `<p>${extra}</p>` : ""}</div>`);
  }
  $("online-list").innerHTML = cards.length
    ? `<div class="card-list">${cards.join("")}</div>`
    : `<div class="report empty-report">No online comparison covers ${name}.</div>`;
}

const SVGNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, text) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
}

const shortNum = (v) => (v >= 1e9 ? `${v / 1e9}G` : v >= 1e6 ? `${v / 1e6}M` : v >= 1e3 ? `${v / 1e3}k` : String(v));

function drawCharts(solve) {
  for (const id of ["chart-changed", "chart-time"]) $(id).replaceChildren();
  if (!solve?.changed?.length) {
    $("chart-changed").innerHTML = '<p class="empty">No per-round record for this material.</p>';
    return;
  }
  const n = solve.changed.length;
  const W = Math.max(320, $("chart-changed").clientWidth), H = 260, P = { l: 52, r: 16, t: 12, b: 28 };
  const pad = 14; // keeps the first and last marks clear of the y labels
  const x = (i) => P.l + pad + (n === 1 ? 0 : (i * (W - P.l - P.r - 2 * pad)) / (n - 1));
  const step = [1, 2, 5, 10, 20, 50].find((s) => (n - 1) / s <= 12) ?? 100;
  const tickRounds = [];
  for (let i = 0; i < n; i += step) tickRounds.push(i);
  const maxV = Math.max(...solve.changed, 1);
  const top = Math.ceil(Math.log10(maxV));
  // Zero sits one decade below 1 so the termination round stays on the chart.
  const y = (v) => H - P.b - ((v <= 0 ? 0 : Math.log10(v) + 1) * (H - P.t - P.b)) / (top + 1);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Words changed per sweep for ${$("evidence-material").value}, log scale` });
  for (let p = 0; p <= top; p += top > 6 ? 2 : 1) {
    svg.append(svgEl("line", { class: "grid", x1: P.l, x2: W - P.r, y1: y(10 ** p), y2: y(10 ** p) }));
    svg.append(svgEl("text", { class: "axis-label", x: P.l - 8, y: y(10 ** p) + 4, "text-anchor": "end" }, shortNum(10 ** p)));
  }
  svg.append(svgEl("text", { class: "axis-label", x: P.l - 8, y: y(0) + 4, "text-anchor": "end" }, "0"));
  for (const i of tickRounds) {
    svg.append(svgEl("text", { class: "axis-label", x: x(i), y: H - 8, "text-anchor": "middle" }, String(i)));
  }
  svg.append(svgEl("path", { class: "series", d: solve.changed.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("") }));
  solve.changed.forEach((v, i) => {
    if (v === 0) svg.append(svgEl("circle", { class: "zero", cx: x(i), cy: y(0), r: 5 }));
  });
  const last = n - 1;
  svg.append(svgEl("text", { class: "axis-label", x: x(last) - 12, y: y(0) + 4, "text-anchor": "end" }, "fixpoint: no word changed"));
  const cross = svgEl("line", { class: "cross", y1: P.t, y2: H - P.b, visibility: "hidden" });
  const dot = svgEl("circle", { class: "dot", r: 5, visibility: "hidden" });
  svg.append(cross, dot);
  const hit = svgEl("rect", { class: "hit", x: P.l - 10, y: 0, width: W - P.l - P.r + 20, height: H });
  svg.append(hit);
  $("chart-changed").append(svg);

  // Sweep time: its own chart (one axis per chart), sharing the round axis.
  const times = solve.sweep_ms.map((v) => (v == null || v < 0 ? 0 : v));
  const H2 = 130;
  const maxT = Math.max(...times, 1);
  const y2 = (v) => H2 - P.b - (v * (H2 - P.t - P.b)) / maxT;
  const svg2 = svgEl("svg", { viewBox: `0 0 ${W} ${H2}`, role: "img", "aria-label": "Sweep time per round" });
  const fmtT = (ms) => (ms === 0 ? "0" : ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s` : `${ms.toFixed(ms < 10 ? 2 : 0)} ms`);
  for (const f of [0, 0.5, 1]) {
    svg2.append(svgEl("line", { class: "grid", x1: P.l, x2: W - P.r, y1: y2(maxT * f), y2: y2(maxT * f) }));
    svg2.append(svgEl("text", { class: "axis-label", x: P.l - 8, y: y2(maxT * f) + 4, "text-anchor": "end" }, fmtT(maxT * f)));
  }
  const bw = Math.max(2, Math.min(18, (W - P.l - P.r - 2 * pad) / n - 2));
  times.forEach((v, i) => {
    const h = Math.max(1, H2 - P.b - y2(v));
    svg2.append(svgEl("rect", { class: "bar", x: x(i) - bw / 2, y: H2 - P.b - h, width: bw, height: h, rx: Math.min(2, bw / 2) }));
  });
  svg2.append(svgEl("line", { class: "grid", x1: P.l, x2: W - P.r, y1: H2 - P.b, y2: H2 - P.b }));
  for (const i of tickRounds) {
    svg2.append(svgEl("text", { class: "axis-label", x: x(i), y: H2 - 8, "text-anchor": "middle" }, String(i)));
  }
  const hit2 = svgEl("rect", { class: "hit", x: P.l - 10, y: 0, width: W - P.l - P.r + 20, height: H2 });
  svg2.append(hit2);
  $("chart-time").append(svg2);

  const tip = $("tooltip");
  const onMove = (e, svgNode) => {
    const box = svgNode.getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - P.l) / (W - P.l - P.r)) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(solve.changed[i])); dot.setAttribute("visibility", "visible");
    const jit = i < 2 ? " (includes JIT compile)" : "";
    tip.innerHTML = `Round ${i}${i === 0 ? " (checkmate seed)" : ""}<br>${fmtInt(solve.changed[i])} words changed<br>sweep ${fmtT(times[i])}${jit}`;
    tip.hidden = false;
    tip.style.left = `${Math.min(e.clientX + 14, innerWidth - 220)}px`;
    tip.style.top = `${e.clientY + 14}px`;
  };
  const onLeave = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); };
  hit.addEventListener("pointermove", (e) => onMove(e, svg));
  hit2.addEventListener("pointermove", (e) => onMove(e, svg2));
  hit.addEventListener("pointerleave", onLeave);
  hit2.addEventListener("pointerleave", onLeave);
}

$("evidence-material").addEventListener("change", renderEvidence);
addEventListener("resize", () => { if (!$("view-evidence").hidden) renderEvidence(); });

// Lock the tools column to the board's height on wide screens (lichess frame).
new ResizeObserver(([entry]) => {
  document.documentElement.style.setProperty("--board-h", `${entry.contentRect.height}px`);
}).observe(document.querySelector(".board-wrap"));

// ---------------------------------------------------------------- boot
async function boot() {
  renderTray();
  [materials, evidence] = await Promise.all([
    fetch("/api/materials").then((r) => r.json()),
    fetch("/api/evidence").then((r) => r.json()),
  ]);
  const options = materials.map((m) => `<option value="${m.material}">${m.material}</option>`).join("");
  $("material").innerHTML = options;
  $("evidence-material").innerHTML = materials.filter((m) => evidence.solves[m.material]).map((m) => `<option>${m.material}</option>`).join("");
  const largest = [...materials].sort((a, b) => b.words - a.words)[0];
  if (largest) { $("material").value = largest.material; $("evidence-material").value = largest.material; }
  renderMaterials();
  const fromHash = decodeURIComponent(location.hash.slice(1));
  await show(parseFen(fromHash) ? fromHash : START, { replace: true });
  const view = new URLSearchParams(location.search).get("view");
  if (["materials", "evidence"].includes(view)) setView(view);
}
boot();
