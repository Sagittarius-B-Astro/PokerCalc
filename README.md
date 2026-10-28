# poker_equity_cpp - Texas Hold'em equity engine (C++) + learned approximation

## Inputs, models, target
**Equity engine (exact / Monte Carlo / hybrid)**
- Inputs: hero's two hole cards, the board (0, 3, 4 or 5 cards), and one range per opponent
  (a list of possible two-card holdings with weights; empty = any two cards).
- Models: none learned. Three estimators of the same probability: exact enumeration, plain Monte Carlo,
  and a stratified "hybrid" (enumerate opponent #1's holdings, sample the rest). Why three: exact is the
  ground truth but explodes; plain MC works for everything; hybrid is exact/cheaper in the cases where
  it can enumerate and has lower variance otherwise (measured below).
- Target: hero's equity = P(win) + share of ties (a k-way tie pays 1/k), each with a standard error.

**Learned approximation (`python/train_approx.py`)**
- Inputs: 62 cheap features of (hole cards, board) computed in `src/features.cpp` (street, ranks, suit and
  straight structure, hero's made-hand category, whether hero improves on the board...).
- Model: four independent fully connected nets 62-128-64-2, one per street (ELU hidden, sigmoid output,
  MSE loss, Adam). The reference paper (arXiv 1808.07220) uses a single 62-128-64-2 model; a per-street
  split was measured to be better here (see "Measured results").
- Target: P(win) and P(tie) against ONE random opponent, labelled by the engine above.

## Files and why they are separate
| File | What it is |
|---|---|
| `src/hand_eval.hpp` | Hand evaluator: up to 7 cards -> integer score (higher wins, equal = exact tie). Header-only so it can be inlined into the inner loop. |
| `src/equity.hpp/.cpp` | The engine: card/range parsing, `exact_equity`, `mc_equity`, `hybrid_equity`. Depends only on the evaluator. |
| `src/features.hpp/.cpp` | Features for the network. Separate because training data generation AND prediction must use the identical function. |
| `src/approx.hpp/.cpp` | Loads trained weights and runs the network. Separate from the engine so the engine never depends on a model file. |
| `src/main.cpp` | Command-line front end (`equity`, `score`, `gen-dataset`, `approx`, `approx-eval`). Only place that prints. |
| `tests/validate.cpp` | Correctness suite. Kept out of `src/` so the library stays free of test/timing code. |
| `python/train_approx.py` | Trains the networks (numpy) and writes `data/approx_weights_s0..s3.bin` (per-street) or `data/approx_weights.bin` (single). |
| `python/crosscheck_treys.py` | Compares the C++ evaluator with the independent `treys` library on 100,000 random hand pairs. |
| `data/` | `approx_weights_s0..s3.bin` (shipped per-street nets, 64 KB each), `approx_weights.bin` (single-model alternative), `test_dataset.csv` (15,000 labelled states), `sample_dataset_preview.csv` (first 500 rows, 6 columns, easy to open), `validation_output.txt`, `approx_eval_output.txt`. |

```
                      hand_eval.hpp  (score 2-7 cards)
                            |
   main.cpp ------------> equity.cpp  (exact / mc / hybrid)
      |  \                      |
      |   \--> features.cpp     +--> labels --> gen-dataset --> train.csv --> train_approx.py
      |            |                                                              |
      |            v                                                              v
      +-------> approx.cpp <-------------- data/approx_weights_s0..s3.bin <------+
   tests/validate.cpp --> equity.cpp + hand_eval.hpp (exhaustive + statistical checks)
```

## Functions
`hand_eval.hpp`: `make_straight_table` builds the straight lookup; `pack_top` packs the best ranks into a score;
`evaluate` scores 2-7 cards (straight flush > quads > full house > flush > straight > trips > two pair > pair > high card).
`equity.cpp`: `parse_card/parse_cards/card_str` text<->cards; `parse_range` reads "QQ+,AKs:0.5";
`for_each_combo` iterates k-subsets; `showdown` gives hero's pot share for one deal; `make_setup` validates inputs and
removes dead cards from ranges; `Picker` draws a weighted holding; `Sampler::draw` deals everything except opponent #1
(whole-combination rejection for ranges); `mc_equity`, `hybrid_equity`, `exact_equity` are the three estimators.
`features.cpp`: `compute_features` builds the 62-number vector; `best_window` counts cards inside the best 5-rank window.
`approx.cpp`: `Net::load` reads weights; `Net::predict` runs the forward pass.
`main.cpp`: one `cmd_*` function per subcommand; `Args`/`parse_args` read `--key value` options; `load_nets` accepts
one weight file or four comma-separated ones (per-street).

## Data: synthetic, and why
No dataset of true equities exists, so labels come from simulating the exact process being modelled (a uniformly
random deal), computed by the engine. Turn and river labels are exact; flop and preflop labels are stratified Monte
Carlo with mean standard error 0.18 points (column `eq_se`), far below the network's error, so it does not limit accuracy.

## Measured results

**Evaluator and estimators (`make test`)**
- All C(52,7) = 133,784,560 seven-card hands classified correctly; every category count matches the published
  table; 4,824 distinct hand strengths (the known number for this ordering).
- Independent cross-check against `treys`: 100,000/100,000 pairwise comparisons agree.
- Plain MC and hybrid are within 4 standard errors of exact enumeration on heads-up and multiway spots, including
  overlapping ranges and k-way ties; reported standard errors match the actual spread within 20% and 95% intervals
  cover the truth 189-194 times out of 200.
- Hybrid is exact on turn and river heads-up; on the flop, its variance reduction vs plain MC at equal showdowns is
  spot-dependent (measured 1.00x on a flush-draw board, 1.30x on a marginal overpair).

**Neural approximation — shipped per-street model**

Four independent nets, architecture 62-128-64-2 each (16,450 parameters, 64.3 KB as float32), trained on
120,000 labelled states with a single 90/10 train/validation split, test set 15,000 rows, seed 2.
Equity MAE on the test set:

| Street   | MAE win | MAE tie | MAE equity |
|----------|--------:|--------:|-----------:|
| preflop  | 0.31%   | 0.11%   | 0.31%      |
| flop     | 1.62%   | 0.75%   | 1.63%      |
| turn     | 2.41%   | 1.03%   | 2.43%      |
| river    | 2.74%   | 1.62%   | 2.85%      |
| overall  | —       | —       | **1.80%**  |

The C++ `approx-eval` reproduces this table exactly, which verifies that the feature computations match between
training and inference.

Comparison with the single-model baseline (62-128-64-2 trained on the same data, seed 2): the single model
reaches 2.03% overall, so per-street is 0.23 points better on the same test set. Per-street's clearest gains are
on preflop (0.31% vs 0.66%) and flop (1.63% vs 2.09%); it ties the single model on river and costs ~0.1 points on
turn. 5-fold cross-validation of the single-model architecture gives mean 2.118% ± 0.202%, so the 0.23 point
margin is comparable to the single-model's fold-to-fold variation: the per-street advantage is real but modest on
the aggregate, and clearest on the early streets. Both models remain well behind the 1,000-showdown MC reference
in overall accuracy (1.10% MAE), which is why the speed comparison below matters.

**Speed — honest comparison**

Measured on the same Windows machine:

| Method | µs / prediction | equity MAE |
|---|---:|---:|
| Per-street network (features + forward pass) | **~5** | 1.80% |
| Plain MC, 1,000 showdowns | 277 | 1.10% |
| Plain MC, ~300 showdowns (≈2% MAE) | ~80 | ~2.03% |

At equal accuracy (~2% MAE), the network is roughly **16x faster** than plain Monte Carlo. The 50x figure often
quoted for neural approximations compares against 1,000-showdown MC, which is *more accurate* than the net; both
numbers are reported so the tradeoff is explicit. For a single high-stakes hand where 2% error is unacceptable,
use MC at 10,000-100,000 showdowns (0.1-0.3% MAE, 3-30 ms) or exact enumeration; the network is not the right
tool for that.

## Run
```
make              # builds ./poker and ./validate
make test         # exhaustive evaluator check + statistical checks (~40 s)
./poker equity --hero AsKs --board Qs7s2d --opp "QQ+,AKs:0.5" --opp random --samples 200000
./poker equity --hero AsKs --board Qs7s2d --method exact
./poker gen-dataset --n 120000 --seed 1 --out train.csv
python python/train_approx.py --train train.csv --test data/test_dataset.csv \
       --per-street --hidden 128 64 --epochs 300 --seed 2
python python/train_approx.py --train train.csv --test data/test_dataset.csv --kfold 5
./poker approx --weights "data/approx_weights_s0.bin,data/approx_weights_s1.bin,data/approx_weights_s2.bin,data/approx_weights_s3.bin" \
               --hero AsKs --board Qs7s2d
./poker approx-eval --weights "data/approx_weights_s0.bin,data/approx_weights_s1.bin,data/approx_weights_s2.bin,data/approx_weights_s3.bin" \
                    --data data/test_dataset.csv
```
Range syntax: `AsKs` exact combo, `QQ` pocket pair, `AKs`/`AKo`/`AK` suited/offsuit/both, `TT+`, `ATs+`,
`:w` relative weight, `random` any hand. Use one `--opp` per opponent.

### `poker approx` / `poker approx-eval` options
- `--weights W` — one PKAP file (single model) or four comma-separated PKAP files, one per street in
  order preflop, flop, turn, river (per-street model).
- `--hero`, `--board` — required for `approx`; ignored by `approx-eval` (which reads them from the CSV).
- `approx-eval` additionally takes `--data FILE`.

### `train_approx.py` options
- `--per-street` — train four independent models, one per street, and export `data/approx_weights_s0..s3.bin`.
- `--kfold K` — K-fold cross-validation on the training set; reports mean / std of held-out MAE per fold.
  No model is exported. Not compatible with `--per-street`.
- `--hidden N M ...` — hidden layer widths (default `128 64`).
- `--warmup N` — linear LR warmup for the first N epochs (default 5).
- `--min-lr LR` — LR floor at the end of the cosine schedule (default 1e-5).
- `--patience N` — stop training if validation MAE has not improved for N epochs (default 50).
- `--out PATH` — weights file base name (default `data/approx_weights.bin`; per-street writes
  `data/approx_weights_s0.bin` ...).
- `--seed S` — RNG seed (default 0).