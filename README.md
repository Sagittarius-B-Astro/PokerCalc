# poker_equity_cpp

A verified Texas Hold'em equity engine (C++), a learned approximation, and a browser-based WebAssembly interface.

[**Source**](https://github.com/Sagittarius-B-Astro/PokerCalc) · [**Web demo**](https://YOUR-SITE.netlify.app) · [**Report**](report.tex) (LaTeX source, compile with `pdflatex`)

---

## Project summary

This web application calculates Texas Hold'em equity through a C++ engine compiled to WebAssembly, exposing exact enumeration, Monte Carlo, and stratified hybrid estimators alongside a per-street neural approximation. Every estimate carries a standard error, and the engine is verified exhaustively against the published 7-card frequency table and statistically against the independent `treys` library. A companion report documents the mathematics, design decisions, and measured results.

---

## Overview

### Inputs and models

**Equity engine (exact / Monte Carlo / hybrid).** Inputs: the hero's two hole cards, the board (0, 3, 4 or 5 cards), and one range per opponent (a list of possible two-card holdings with weights; empty = any two cards). Three estimators of the same probability: exact enumeration (ground truth but explodes), plain Monte Carlo (works for everything), and stratified hybrid (enumerates opponent #1's holdings and samples the rest). Every estimator reports a standard error.

**Learned approximation.** Input: 62 cheap features of (hero hole cards, board) computed in `src/features.cpp`. Model: four independent fully connected networks, 62-128-64-2, one per street (ELU hidden, sigmoid output, MSE loss, Adam). Target: P(win) and P(tie) vs. one random opponent, labelled by the engine above.

**Target.** Hero's equity = P(win) + share of ties (a k-way tie pays 1/k), each with a standard error.

### What runs where

| Layer | Native binary (`poker.exe`) | WebAssembly (`docs/poker.wasm`) |
|---|---|---|
| Hand evaluator | Yes | Yes |
| Exact / MC / hybrid estimators | Yes | Yes |
| Neural approximation | Yes | Yes |
| CLI commands (`equity`, `score`, `gen-dataset`, `approx`, `approx-eval`) | Yes | No — web exposes functions directly |
| Dataset generation | Yes | Not exposed to JS |
| Training (`train_approx.py`) | Offline only | No |

The WebAssembly build reuses `equity.cpp`, `features.cpp`, `approx.cpp`, and `hand_eval.hpp` unchanged. Only `src/wasm_bindings.cpp` is new, and it is not compiled into the native binary.

---

## Files

### Library and CLI

| File | Purpose |
|---|---|
| `src/hand_eval.hpp` | Hand evaluator: up to 7 cards → integer score (higher wins, equal = exact tie). Header-only so it inlines into the inner loop. |
| `src/equity.hpp/.cpp` | The engine: card/range parsing, `exact_equity`, `mc_equity`, `hybrid_equity`. Depends only on the evaluator. |
| `src/features.hpp/.cpp` | The 62-number feature vector. Shared by training data generation and inference — identical function in both paths. |
| `src/approx.hpp/.cpp` | Loads trained weights and runs the forward pass. Includes `kMaxLayerWidth = 256` guard in `Net::load` (rejects too-wide models with a clear error). |
| `src/main.cpp` | CLI front end (`equity`, `score`, `gen-dataset`, `approx`, `approx-eval`). Only place that prints to stdout in the native build. |
| `src/wasm_bindings.cpp` | Emscripten `EMSCRIPTEN_BINDINGS` glue. Exposes `exactEquity`, `mcEquity`, `hybridEquity`, `approx` to JavaScript. Not compiled in the native build. |

### Web (browser)

| File | Purpose |
|---|---|
| `docs/index.html` | The UI: header with GitHub link, settings panel, poker table, output panel. |
| `docs/style.css` | Dark theme, poker-felt table, card rendering. |
| `docs/app.js` | Main-thread controller: DOM, validation, worker bridge, cancel button. |
| `docs/worker.js` | Web Worker: loads the Emscripten module, runs all heavy computation off the main thread. |
| `docs/poker.js` | Emscripten-generated JS glue (built by `make wasm`, committed). |
| `docs/poker.wasm` | Emscripten-generated WASM binary (built by `make wasm`, committed). |
| `docs/poker.data` | Emscripten-generated preload package containing the five weight files (~320 KB, built by `make wasm`, committed). |
| `netlify.toml` | `publish = "docs"`, `command = ""`. Deploys the built site from `docs/`. |

### Training and evaluation scripts

| File | Purpose |
|---|---|
| `python/train_approx.py` | Trains the networks (numpy) and writes `data/approx_weights_s0..s3.bin` (per street) or `data/approx_weights.bin` (single). Supports `--per-street`, `--kfold`, `--hidden`, `--warmup`, `--min-lr`, `--patience`. |
| `python/crosscheck_treys.py` | Compares the C++ evaluator with the independent `treys` library on 100,000 random pairs. |
| `figures/make_figures.py` | Generates the four report figures from `data/test_dataset.csv` and the shipped weights. |

### Tests and verification

| File | Purpose |
|---|---|
| `tests/validate.cpp` | Correctness suite: exhaustive evaluator check, hand-written ordering checks, statistical coverage, variance reduction. Kept out of `src/` so the library stays free of test code. |

### Documentation

| File | Purpose |
|---|---|
| `report.tex` | LaTeX companion report: goals, mathematics, design decisions, measured results, summary. |
| `Figures/fig_*.pdf` | Report figures, generated by `figures/make_figures.py`. |

### Data

| Path | Purpose |
|---|---|
| `data/approx_weights_s0..s3.bin` | Shipped per-street neural nets (64.3 KB each, ~256 KB total). |
| `data/approx_weights.bin` | Alternative single-model weights (~64 KB). |
| `data/approx_weights.bin.report.txt` | Training report for the single-model baseline. |
| `data/test_dataset.csv` | 15,000 labelled held-out states. |
| `data/sample_dataset_preview.csv` | First 500 rows, 6 columns, easy to open. |
| `data/validation_output.txt` | Captured output of `./validate.exe` (all checks passing). |
| `data/approx_eval_output.txt` | Captured output of `./poker approx-eval` on the test set. |

### Deployment

`docs/` is committed, including the WASM artifacts. Netlify's build container does not have Emscripten installed, so `make wasm` must be run locally before pushing.

---

## Architecture

```
                       hand_eval.hpp  (score 2-7 cards)
                            |
      main.cpp ----------> equity.cpp  (exact / mc / hybrid)
         |  \                  |
         |   \--> features.cpp |
         |            |        +--> labels --> gen-dataset --> train.csv --> train_approx.py
         |            |                                                           |
         |            v                                                           v
         +-------> approx.cpp <------------ data/approx_weights_s0..s3.bin <-----+
                       |
         wasm_bindings.cpp --+
                            |
                            +--> docs/poker.{js,wasm,data}  (Emscripten output)
                            |
                            +--> docs/worker.js             (Web Worker)
                            |
                            +--> docs/app.js                (main thread)
                            |
                            +--> docs/index.html            (served by Netlify)

      tests/validate.cpp --> equity.cpp + hand_eval.hpp  (exhaustive + statistical checks)
```

---

## Functions

`hand_eval.hpp`: `make_straight_table` builds the 8,192-entry straight lookup (ace counted both high and low); `pack_top` packs the best ranks into a score; `evaluate` scores 2-7 cards (straight flush > quads > full house > flush > straight > trips > two pair > pair > high card).

`equity.cpp`: `parse_card` / `parse_cards` / `card_str` handle text↔cards; `parse_range` reads `"QQ+,AKs:0.5"`; `for_each_combo` iterates k-subsets; `showdown` computes hero's pot share for one deal, handling k-way ties; `make_setup` validates inputs and removes dead cards from ranges; `Picker` draws weighted holdings; `Sampler::draw` deals everything except opponent #1 (whole-combination rejection); `mc_equity`, `hybrid_equity`, `exact_equity` are the three estimators.

`features.cpp`: `compute_features` builds the 62-vector; `best_window` counts distinct ranks inside the best 5-rank window.

`approx.cpp`: `Net::load` reads the PKAP binary, enforces `kMaxLayerWidth = 256`; `Net::predict` runs the forward pass.

`wasm_bindings.cpp`: exposes `exactEquity`, `mcEquity`, `hybridEquity`, `approx` to JavaScript via `EMSCRIPTEN_BINDINGS`. Ranges are joined with `|` (individual ranges use `,` and `:` internally).

`main.cpp`: one `cmd_*` function per subcommand; `Args` / `parse_args` read `--key value` options; `load_nets` accepts one weight file or four comma-separated per-street files.

`python/train_approx.py`: `load_csv`, `MLP` (manual backprop), `Adam`, `train_one`, `report`, `export_weights` (PKAP writer), `main`.

---

## Data

No dataset of true equities exists, because equity can only be computed, not measured. Labels are generated by the engine itself: a uniformly random deal at the chosen street, computed by the estimator appropriate to the leaf count.

- **Turn and river labels** are exact (`hybrid_equity` enumerates every runout when the count per holding is small enough).
- **Preflop and flop labels** are stratified Monte Carlo with a reported per-row standard error (`eq_se` column of the training CSV). The mean label SE is ~0.18 points on preflop/flop, far below the network's own error, so label noise does not bind accuracy.

Training budget per street: preflop 12 attempts per opponent holding (14,700 samples total), flop 12 (12,972 total), turn 45 (exact — every runout visited), river 2 (exact — one runout per holding).

---

## Measured results

### Evaluator and estimators (`make test`)

- All C(52,7) = 133,784,560 seven-card hands classified correctly; category counts match the published table; 4,824 distinct hand strengths (the known value for this ordering).
- Independent cross-check against `treys`: 100,000/100,000 pairwise comparisons agree.
- Plain MC and hybrid are within 4 standard errors of exact enumeration on heads-up and multiway spots, including overlapping ranges and k-way ties; reported standard errors match the actual empirical spread within 20%, and 95% intervals cover the truth 189–194 times out of 200.
- Hybrid is exact on turn and river heads-up. On the flop, its variance reduction vs. plain MC at equal showdowns is spot-dependent (measured 1.00× on a flush-draw board, 1.30× on a marginal overpair).

### Neural approximation — shipped per-street model

Four independent nets, architecture 62-128-64-2 each (16,450 parameters, 64.3 KB as float32), trained on 120,000 labelled states with a single 90/10 train/validation split, test set 15,000 rows, seed 2. Equity MAE on the test set:

| Street | MAE win | MAE tie | MAE equity |
|---|---:|---:|---:|
| preflop | 0.31% | 0.11% | 0.31% |
| flop | 1.62% | 0.75% | 1.63% |
| turn | 2.41% | 1.03% | 2.43% |
| river | 2.74% | 1.62% | 2.85% |
| **overall** | — | — | **1.80%** |

The C++ `approx-eval` reproduces this table exactly. The web `approx` function matches the CLI `approx` command to the last printed digit, confirming the feature computations are identical between training, native inference, and WASM inference.

Single-model baseline (62-128-64-2 trained on the same data, seed 2) reaches 2.03% overall. Per-street is 0.23 points better. 5-fold cross-validation of the single-model architecture gives mean 2.118% ± 0.202%, so the 0.23-point margin is comparable to the single-model's fold-to-fold variation: the per-street advantage is real but modest, and clearest on preflop (0.31% vs 0.66%) and flop (1.63% vs 2.09%). Both models remain behind the 1,000-showdown MC reference in overall accuracy (1.10% MAE).

### Speed — honest comparison

Measured on the same Windows machine:

| Method | µs / prediction | Equity MAE |
|---|---:|---:|
| Per-street network (features + forward pass) | **~5** | 1.80% |
| Plain MC, 1,000 showdowns | 277 | 1.10% |
| Plain MC, ~300 showdowns (≈2% MAE) | ~80 | ~2.03% |

At equal accuracy (~2% MAE), the network is roughly **16×** faster than plain Monte Carlo. The 50× figure often quoted for neural approximations compares against 1,000-showdown MC, which is *more accurate* than the net; both numbers are reported so the tradeoff is explicit. For a single high-stakes hand where 2% error is unacceptable, use MC at 10,000–100,000 showdowns (0.1–0.3% MAE, 3–30 ms) or exact enumeration; the network is not the right tool for that.

The WASM build adds no measurable overhead on top of native (the neural forward pass is ~5 µs; the WASM-to-JS call boundary adds ~1 µs; the total is dominated by the forward pass). Monte Carlo and exact enumeration run at roughly 0.7× native speed in WASM, which is typical for compute-bound workloads under Emscripten's `-O3`.

---

## Verification

Three independent checks, run locally.

### `make test` (~40 s)

The `validate` binary:

- **Part A** — exhaustive evaluator check: all 133,784,560 seven-card hands, category counts match the published table, 4,824 distinct strengths.
- **Part B** — hand-written ordering checks: wheel vs. six-high straight, flush vs. full house, kickers within a category, three-pairs-as-two-pair, quads-plus-trips-as-quads.
- **Part C** — estimators vs. exact: heads-up river (hybrid must be exact), heads-up turn, heads-up flop, 2-opponent river, overlapping ranges with card removal, 3-way tie on a royal-flush board.
- **Part D** — standard-error honesty: 200 repeats per estimator, comparing reported SE to actual spread, and 95%-interval coverage. Also compares hybrid variance reduction against plain MC at equal showdown counts.

Exit code is 0 iff all checks pass.

### `crosscheck_treys.py`

```bash
pip install treys
python python/crosscheck_treys.py ./poker.exe 100000
# expected: 100000/100000 pairwise comparisons agree with treys
```

This is the only check that uses an **independent** evaluator. `treys` uses a different scoring scheme (lower is better) and a different lookup implementation, so agreement on order is strong evidence the C++ evaluator is correct beyond just matching its own frequency table.

### `approx-eval`

```bash
./poker.exe approx-eval \
  --weights "data/approx_weights_s0.bin,data/approx_weights_s1.bin,data/approx_weights_s2.bin,data/approx_weights_s3.bin" \
  --data data/test_dataset.csv
```

Prints the same per-street MAE table as above. If this disagrees with the README, the weights on disk or the test set on disk differ from what was measured.

---

## Building

### Native (Linux / macOS / Windows with `g++` or MSVC)

```bash
make                # builds ./poker and ./validate
make test           # exhaustive evaluator check + statistical checks (~40 s)
```

On Windows without `make`, compile directly:

```powershell
g++ -O2 -std=c++17 -I src src/main.cpp src/equity.cpp src/features.cpp src/approx.cpp -o poker.exe
g++ -O2 -std=c++17 -I src tests/validate.cpp src/equity.cpp src/features.cpp src/approx.cpp -o validate.exe
```

The native build never sees `src/wasm_bindings.cpp` (its Makefile target lists sources explicitly).

### WebAssembly (requires Emscripten)

The WASM target uses **`em++`**, not `emcc`. Using `emcc` fails at the link step with undefined C++ standard-library symbols (`std::invalid_argument`, `__cxa_throw`, `operator delete`, and similar). `em++` links `libc++` automatically.

**Install Emscripten** (once):

```powershell
# Windows (from the emsdk folder)
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
.\emsdk.bat install latest
.\emsdk.bat activate latest
.\emsdk_env.bat          # or use emsdk_env.ps1 in PowerShell to persist env
```

The `emsdk/` directory is not tracked (see `.gitignore`); it is a 1-2 GB toolchain install and should stay local.

**Build the WASM** (from the project root, with `emcc`/`em++` on PATH):

```bash
make wasm
```

which runs:

```bash
em++ -O3 -std=c++17 -I src \
  src/equity.cpp src/features.cpp src/approx.cpp src/wasm_bindings.cpp \
  --bind \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=createPokerModule \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s ENVIRONMENT=web,worker \
  --preload-file data/approx_weights_s0.bin \
  --preload-file data/approx_weights_s1.bin \
  --preload-file data/approx_weights_s2.bin \
  --preload-file data/approx_weights_s3.bin \
  --preload-file data/approx_weights.bin \
  -o docs/poker.js
```

Output lands in `docs/`: `poker.js`, `poker.wasm`, `poker.data`. The `--preload-file` flags bundle the weight files into `poker.data`, so `Net::load("data/approx_weights_s0.bin")` works unchanged inside the WASM virtual filesystem.

**Verify** the build produced all three files with non-zero size:

```powershell
Get-ChildItem docs\poker.* | Select-Object Name, Length
# poker.js    ~50 KB
# poker.wasm  ~500 KB
# poker.data  ~320 KB
```

---

## Running

### CLI

```bash
# Heads-up, flop, hybrid estimator (the CLI default)
./poker equity --hero AsKs --board Qs7s2d --opp random --samples 200000

# Exact enumeration, heads-up flop
./poker equity --hero AsKs --board Qs7s2d --method exact

# Overlapping ranges with weights
./poker equity --hero AsKs --board Qs7s2d --opp "QQ+,AKs:0.5" --opp random --samples 200000

# Neural approximation (per-street weights)
./poker approx --weights "data/approx_weights_s0.bin,data/approx_weights_s1.bin,data/approx_weights_s2.bin,data/approx_weights_s3.bin" --hero AsKs --board Qs7s2d

# Regenerate the training set (120,000 rows)
./poker gen-dataset --n 120000 --seed 1 --out train.csv

# Retrain (per-street, the shipped configuration)
python python/train_approx.py --train train.csv --test data/test_dataset.csv \
       --per-street --hidden 128 64 --epochs 300 --seed 2

# 5-fold cross-validation (single-model architecture, report only, no export)
python python/train_approx.py --train train.csv --test data/test_dataset.csv --kfold 5
```

### Local web

```bash
python -m http.server 8000 --directory docs
# open http://localhost:8000/
```

The trailing-slash rule matters: `python -m http.server --directory docs` makes `docs/` the server root, so `http://localhost:8000/style.css` maps to `docs/style.css` and `./style.css` inside the HTML resolves correctly. Serving from the project root without `--directory` and opening `/` serves the wrong directory and 404s on the CSS/JS.

### Deployed web

The site is deployed at the Netlify URL at the top of this README. The site is fully static (no backend): `docs/index.html` loads `docs/app.js`, which spins up `docs/worker.js`, which loads `docs/poker.js` / `poker.wasm` / `poker.data`, all served from the same origin.

---

## Deployment

`netlify.toml` at the project root:

```toml
[build]
  publish = "docs"
  command = ""
```

- `publish = "docs"` tells Netlify to serve the `docs/` folder as the site root.
- `command = ""` tells Netlify not to run a build — the files are built locally by `make wasm` and committed.

**Setup, once:**

1. Push `docs/` including `poker.js`, `poker.wasm`, `poker.data` to GitHub. Netlify's build container does not have Emscripten installed, so those files must be committed.
2. Connect the GitHub repo to Netlify via **Add new site → Import an existing project**.
3. Netlify reads `netlify.toml` and pre-fills the publish directory and build command. Confirm both match.
4. Deploy.

**Ongoing:** every `git push` to `main` triggers a redeploy. When you change the C++ and rebuild the WASM, commit the updated `docs/poker.*` alongside the source changes.

**Note on repo visibility:** Netlify can deploy from a private GitHub repo, so the source can stay private while the deployed site is public. This project's repo is public as a portfolio piece, but the `publish = "docs"` setting keeps the deployed root cleanly separated from the source tree.

---

## Companion report and figures

`report.tex` is a full LaTeX writeup of the project: goals, mathematics, design decisions, measured results, summary, and appendices containing the leaf-count proof, the delta-method variance proof, and selected code. Compile with:

```bash
pdflatex report.tex
pdflatex report.tex       # second pass for the table of contents
```

Figures live in `Figures/`. Regenerate them with:

```bash
pip install matplotlib numpy
python figures/make_figures.py --out-dir Figures
```

The script reads `data/test_dataset.csv` and the shipped weights, and prints the per-street MAE table at the end so you can cross-check against the README's table.

---

## Range syntax reference

| Token | Meaning |
|---|---|
| `AsKs` | One exact combo |
| `QQ` | All pocket queens |
| `AKs` / `AKo` / `AK` | Suited / offsuit / both |
| `TT+` | Pairs TT through AA |
| `ATs+` | ATs, AJs, AQs, AKs |
| `:w` | Relative weight (e.g., `QQ:0.5`) |
| `random` / `any` | Any two cards |
| `|` (WASM only) | Separates multiple opponent ranges when passed through the JS binding |

In the CLI, use one `--opp` flag per opponent. In the browser, use the "Add opponent" button.

---

## Known limitations

- **The neural approximation is heads-up only.** It is trained against one random opponent. Multiway states route through the exact / MC / hybrid estimators; there is no neural shortcut.
- **Preflop and multiway exact enumeration is not feasible in the browser.** The UI's exact cap is 50,000,000 leaves (covering heads-up through flop, and two-opponent turn/river). Preflop-exact is disabled in the dropdown. The CLI supports a larger cap on slower hardware.
- **Neural accuracy on river is 2.85% MAE.** For decisions where that is unacceptable, use MC at 10,000+ samples or exact enumeration (river heads-up is exact in microseconds).
- **The WASM build is synchronous in a Worker but not preemptively cancelable mid-call.** Cancel terminates the worker and restarts it, which is correct for the ~40 M-leaf exact cases but not instantaneous. Deep exact runs (2-opponent turn) may take several seconds before cancel can take effect.

---

## Development notes (not tracked)

The following exist locally but are ignored by `.gitignore`:

- `emsdk/` — Emscripten toolchain (~1-2 GB; install per machine)
- `venv/` — Python virtual environment
- `poker.exe`, `validate.exe` — native build outputs
- `train.csv` — regenerable training set (~120,000 rows)
- `data/wide.bin`, `data/wide.bin.report.txt` — test artifacts from a wider-model experiment
- `.vscode/` — editor config

To rebuild everything after a fresh clone:

```bash
# Prerequisites: g++, Python 3, Emscripten (if building WASM)
python -m venv venv
source venv/bin/activate       # Windows: .\venv\Scripts\Activate.ps1
pip install numpy treys matplotlib

make                            # native
make wasm                       # WebAssembly (requires emcc/em++ on PATH)

make test                       # verify
python python/crosscheck_treys.py ./poker.exe 100000   # if treys installed
```