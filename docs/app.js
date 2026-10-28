/* app.js — main thread.
 * Owns the DOM and message-passes to worker.js for all heavy computation.
 * Flip PREVIEW_MODE to true to test the UI without a WASM build.
 */
'use strict';

const PREVIEW_MODE = false;
const MAX_EXACT_LEAVES = 50_000_000;
const MULTIWAY_WARN_THRESHOLD = 2;   // 2+ opponents => show warning

/* ============================================================ DOM */
const $ = id => document.getElementById(id);
const actionSel     = $('action');
const actionHint    = $('action-hint');
const heroInput     = $('hero');
const boardInput    = $('board');
const oppList       = $('opp-list');
const addOppBtn     = $('add-opp');
const multiwayWarn  = $('multiway-warning');
const samplesRow    = $('samples-row'),  samplesInput = $('samples');
const seedRow       = $('seed-row'),     seedInput    = $('seed');
const weightsRow    = $('weights-row'),  weightsInput = $('weights');
const calcBtn       = $('calc');
const cancelBtn     = $('cancel');
const statusEl      = $('status');
const outputEl      = $('output');
const heroCardsEl   = $('hero-cards');
const boardCardsEl  = $('board-cards');
const opponentsEl   = $('opponents');

/* ============================================================ Card render */
const SUIT_GLYPH = { s: '♠', h: '♥', d: '♦', c: '♣' };
const CARD_RE = /^[2-9TJQKA][shdc]$/i;

function parseCards(str) {
  str = (str || '').replace(/\s+/g, '').toUpperCase();
  if (str.length % 2) return null;
  const out = [];
  for (let i = 0; i < str.length; i += 2) {
    const c = str.slice(i, i + 2);
    if (!CARD_RE.test(c)) return null;
    out.push(c);
  }
  return out;
}

function cardEl(cs) {
  const rank = cs[0], suit = cs[1].toLowerCase();
  const el = document.createElement('div');
  el.className = 'card' + ((suit === 'h' || suit === 'd') ? ' red' : '');
  el.innerHTML = `<span class="rank">${rank}</span><span class="suit">${SUIT_GLYPH[suit]}</span>`;
  return el;
}
function emptyCardEl() { const d = document.createElement('div'); d.className = 'card empty'; return d; }
function backCardEl()  { const d = document.createElement('div'); d.className = 'card back'; return d; }

function renderHero() {
  heroCardsEl.innerHTML = '';
  const c = parseCards(heroInput.value);
  if (c && c.length === 2) c.forEach(x => heroCardsEl.appendChild(cardEl(x)));
  else { heroCardsEl.appendChild(emptyCardEl()); heroCardsEl.appendChild(emptyCardEl()); }
}
function renderBoard() {
  boardCardsEl.innerHTML = '';
  const c = parseCards(boardInput.value);
  const n = c ? Math.min(c.length, 5) : 0;
  for (let i = 0; i < 5; i++) boardCardsEl.appendChild(i < n ? cardEl(c[i]) : emptyCardEl());
}
function renderOpponents() {
  opponentsEl.innerHTML = '';
  const rows = [...document.querySelectorAll('.opp-input')];
  const specs = rows.length ? rows.map(i => i.value.trim() || 'random') : ['random'];
  for (const spec of specs) {
    const wrap = document.createElement('div'); wrap.className = 'opp-stack';
    const cards = document.createElement('div'); cards.className = 'cards';
    cards.appendChild(backCardEl()); cards.appendChild(backCardEl());
    const label = document.createElement('div'); label.className = 'label'; label.textContent = spec;
    wrap.appendChild(cards); wrap.appendChild(label);
    opponentsEl.appendChild(wrap);
  }
}

/* ============================================================ Opponent list */
function addOppRow(value = '') {
  const row = document.createElement('div'); row.className = 'opp-row';
  const input = document.createElement('input');
  input.className = 'opp-input'; input.type = 'text';
  input.placeholder = 'random'; input.value = value;
  input.spellcheck = false; input.autocomplete = 'off';
  input.addEventListener('input', () => { renderOpponents(); syncWarnings(); });
  const rm = document.createElement('button');
  rm.type = 'button'; rm.textContent = '×'; rm.title = 'Remove';
  rm.addEventListener('click', () => { row.remove(); renderOpponents(); syncWarnings(); });
  row.appendChild(input); row.appendChild(rm);
  oppList.appendChild(row);
  return input;
}

/* ============================================================ Status */
function setStatus(msg, kind /* 'info' | 'error' | 'busy' */) {
  statusEl.textContent = msg;
  statusEl.hidden = false;
  statusEl.className = 'status' + (kind === 'error' ? ' error' : kind === 'busy' ? ' busy' : '');
}
function clearStatus() { statusEl.hidden = true; statusEl.textContent = ''; }
function setBusy(busy) {
  calcBtn.disabled = busy;
  cancelBtn.hidden = !busy;
  calcBtn.textContent = busy ? 'Calculating…' : 'Calculate';
}

/* ============================================================ Action visibility + rules */
function syncActionVisibility() {
  const a = actionSel.value;
  samplesRow.hidden = (a === 'exact' || a === 'approx');
  seedRow.hidden    = (a === 'exact' || a === 'approx');
  weightsRow.hidden = (a !== 'approx');
}

function syncWarnings() {
  // Disable "exact" for preflop (empty board): always too large.
  const boardCards = parseCards(boardInput.value);
  const preflop = boardCards !== null && boardCards.length === 0;
  const exactOpt = actionSel.querySelector('option[value="exact"]');
  exactOpt.disabled = preflop;
  if (preflop && actionSel.value === 'exact') {
    actionSel.value = 'hybrid';
    syncActionVisibility();
  }
  if (preflop) {
    actionHint.textContent = 'Exact is disabled preflop (≈2.1 billion leaves).';
    actionHint.hidden = false;
  } else {
    actionHint.hidden = true;
  }

  // Multiway warning.
  const oppCount = document.querySelectorAll('.opp-input').length;
  if (oppCount >= MULTIWAY_WARN_THRESHOLD) {
    multiwayWarn.hidden = false;
    multiwayWarn.textContent =
      `With ${oppCount} opponents, the calculation tree grows very fast — ` +
      (actionSel.value === 'exact'
        ? 'exact may fail; consider hybrid or Monte Carlo.'
        : 'expect longer run times. Consider fewer samples or hybrid.');
  } else {
    multiwayWarn.hidden = true;
  }
}

/* ============================================================ Output render */
function renderError(msg) {
  outputEl.innerHTML = `<div class="output-empty" style="color:#fca5a5">${msg}</div>`;
}

function renderResult(r, dt) {
  const isApprox = typeof r.win === 'number';
  const win = isApprox ? { value: r.win, se: null } : r.win;
  const tie = isApprox ? { value: r.tie, se: null } : r.tie;
  const eq  = isApprox ? { value: r.equity, se: null } : r.equity;
  const pct = v => (100 * v).toFixed(2) + '%';
  const se  = s => s == null ? '' : `± ${(100 * s).toFixed(3)}`;
  const methodLabel = isApprox
    ? 'Network approximation'
    : r.exact
      ? 'Exact enumeration'
      : (r.samples ? `Monte Carlo / hybrid · ${r.samples.toLocaleString()} samples` : 'Estimate');
  outputEl.innerHTML = `
    <div class="result-header">
      <span class="method">${methodLabel}</span>
      <span class="meta">computed in ${dt.toFixed(1)} ms</span>
    </div>
    <div class="stats">
      <div class="stat"><span class="label">Win</span><span class="value">${pct(win.value)}</span><span class="se">${se(win.se)}</span></div>
      <div class="stat"><span class="label">Tie</span><span class="value">${pct(tie.value)}</span><span class="se">${se(tie.se)}</span></div>
      <div class="stat accent"><span class="label">Equity</span><span class="value">${pct(eq.value)}</span><span class="se">${se(eq.se)}</span></div>
    </div>
    <div class="bar"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, 100 * eq.value))}%"></div></div>`;
}

/* ============================================================ Worker bridge */
let worker = null;
let workerReady = false;
let nextId = 0;
const pending = new Map();  // id -> { resolve, reject }

function initWorker() {
  if (PREVIEW_MODE) {
    workerReady = true;
    setStatus('Preview mode — results are fake numbers.', 'busy');
    return;
  }
  try {
    worker = new Worker('./worker.js');
  } catch (e) {
    setStatus('Could not start worker: ' + e.message, 'error');
    return;
  }
  worker.onmessage = (ev) => {
    const d = ev.data || {};
    if (d.type === 'ready') {
      workerReady = true;
      setStatus('Engine ready.', 'info');
      return;
    }
    if (d.type === 'boot-error') {
      workerReady = false;
      setStatus(d.message || 'Engine failed to load.', 'error');
      return;
    }
    const req = pending.get(d.id);
    if (!req) return;
    pending.delete(d.id);
    if (d.type === 'result') req.resolve({ result: d.result, dt: d.dt });
    else req.reject(new Error(d.message || 'Unknown error'));
  };
  worker.onerror = (e) => {
    workerReady = false;
    setStatus('Worker error: ' + (e.message || 'unknown'), 'error');
  };
}

function terminateWorker() {
  if (worker) { worker.terminate(); worker = null; }
  workerReady = false;
  pending.forEach(r => r.reject(new Error('cancelled')));
  pending.clear();
}

function callWorker(payload) {
  if (PREVIEW_MODE) return Promise.resolve(fakeCall(payload));
  if (!workerReady) return Promise.reject(new Error('Engine not ready'));
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, ...payload });
  });
}

function fakeCall({ method, samples }) {
  const w = 0.55 + Math.random() * 0.20;
  const t = 0.01 + Math.random() * 0.04;
  const isExact = method === 'exact';
  const isApprox = method === 'approx';
  if (isApprox) return { result: { win: w, tie: t, equity: w + t / 2 }, dt: 0.005 };
  const se = isExact ? 0 : 0.005;
  return {
    result: {
      win:    { value: w, se },
      tie:    { value: t, se },
      equity: { value: w + t / 2, se },
      samples: isExact ? 1_070_190 : (samples || 200000),
      exact:   isExact,
    },
    dt: isExact ? 12 : 50,
  };
}

/* ============================================================ Calculate */
async function calculate() {
  if (!workerReady) {
    renderError('Engine not ready. Build docs/poker.js (see Makefile `wasm` target).');
    return;
  }

  const method = actionSel.value;
  const hero   = heroInput.value.replace(/\s+/g, '');
  const board  = boardInput.value.replace(/\s+/g, '');
  const opps   = [...document.querySelectorAll('.opp-input')]
                  .map(i => i.value.trim()).filter(Boolean);

  const errs = [];
  const heroCards = parseCards(hero);
  if (!heroCards || heroCards.length !== 2) errs.push('Hero needs exactly 2 valid cards.');
  const boardCards = parseCards(board);
  if (boardCards === null) errs.push('Board contains an invalid card.');
  else if (![0, 3, 4, 5].includes(boardCards.length))
    errs.push('Board needs 0, 3, 4 or 5 cards.');
  if (errs.length) { renderError(errs.join(' ')); return; }

  const oppSpecs = opps.length ? opps : ['random'];
  const samples  = Math.max(2, parseInt(samplesInput.value, 10) || 200000);
  const seed     = parseInt(seedInput.value, 10) || 1;
  const weights  = weightsInput.value.trim();

  setBusy(true);
  setStatus('Calculating…', 'busy');
  // let the browser paint the "Calculating…" state before we hand off
  await new Promise(r => setTimeout(r, 16));

  try {
    const { result, dt } = await callWorker({
      method, hero, board, opps: oppSpecs,
      samples, seed, weights, maxLeaves: MAX_EXACT_LEAVES,
    });
    renderResult(result, dt);
    setStatus(`Done — ${dt.toFixed(0)} ms.`, 'info');
  } catch (e) {
    if (e.message === 'cancelled') return;  // don't render anything on cancel
    renderError(e.message || String(e));
    setStatus('Calculation failed.', 'error');
  } finally {
    setBusy(false);
  }
}

function cancelCalculate() {
  terminateWorker();
  setStatus('Cancelled. Restarting engine…', 'busy');
  initWorker();
}

/* ============================================================ Wiring */
heroInput.addEventListener('input', renderHero);
boardInput.addEventListener('input', () => { renderBoard(); syncWarnings(); });
actionSel.addEventListener('change', () => { syncActionVisibility(); syncWarnings(); });
calcBtn.addEventListener('click', calculate);
cancelBtn.addEventListener('click', cancelCalculate);
addOppBtn.addEventListener('click', () => {
  const inp = addOppRow('');
  inp.focus();
  syncWarnings();
  renderOpponents();
});
[heroInput, boardInput, samplesInput, seedInput].forEach(el => {
  el.addEventListener('keydown', e => { if (e.key === 'Enter') calculate(); });
});

/* ============================================================ Boot */
addOppRow('random');
renderHero(); renderBoard(); renderOpponents(); syncActionVisibility(); syncWarnings();
initWorker();