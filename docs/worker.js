/* worker.js — runs all heavy computation off the main thread.
 * Loads the Emscripten module (poker.js, produced by `make wasm`),
 * exposes it to the main thread via postMessage, and never blocks the UI.
 */
'use strict';

importScripts('./poker.js');   // defines self.createPokerModule

let engine = null;

createPokerModule({
  // Emscripten resolves `poker.wasm` and `poker.data` relative to the worker
  // script by default; being explicit here is cheap insurance.
  locateFile: (path) => new URL(path, self.location.href).href,
}).then(Module => {
  engine = {
    exact:  (hero, board, opps, maxLeaves) =>
      Module.exactEquity(hero, board, opps.join('|'), maxLeaves),
    mc:     (hero, board, opps, samples, seed) =>
      Module.mcEquity(hero, board, opps.join('|'), samples, seed),
    hybrid: (hero, board, opps, samples, seed) =>
      Module.hybridEquity(hero, board, opps.join('|'), samples, seed),
    approx: (weights, hero, board) =>
      Module.approx(weights, hero, board),
  };
  self.postMessage({ type: 'ready' });
}).catch(err => {
  self.postMessage({
    type: 'boot-error',
    message: 'Engine failed to load: ' + (err && err.message ? err.message : String(err)),
  });
});

self.onmessage = (ev) => {
  const d = ev.data;
  if (!d || !d.id) return;
  if (!engine) {
    self.postMessage({ id: d.id, type: 'error', message: 'Engine not ready' });
    return;
  }
  try {
    const t0 = performance.now();
    let result;
    if      (d.method === 'exact')  result = engine.exact(d.hero, d.board, d.opps, d.maxLeaves);
    else if (d.method === 'mc')     result = engine.mc(d.hero, d.board, d.opps, d.samples, d.seed);
    else if (d.method === 'hybrid') result = engine.hybrid(d.hero, d.board, d.opps, d.samples, d.seed);
    else if (d.method === 'approx') result = engine.approx(d.weights, d.hero, d.board);
    else throw new Error('unknown method: ' + d.method);
    const dt = performance.now() - t0;
    self.postMessage({ id: d.id, type: 'result', result, dt });
  } catch (err) {
    self.postMessage({ id: d.id, type: 'error', message: err.message || String(err) });
  }
};