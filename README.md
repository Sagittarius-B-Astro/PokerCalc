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
- Model: fully connected net 62-128-64-2 (ELU hidden, sigmoid output, MSE loss, Adam) - the recipe of the
  reference paper (arXiv 1808.07220), a bit wider because our feature vector is larger.
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
| `python/train_approx.py` | Trains the network (numpy) and writes `data/approx_weights.bin`. Python because training is exploratory; inference stays in C++. |
| `python/crosscheck_treys.py` | Compares the C++ evaluator with the independent `treys` library on 100,000 random hand pairs. |
| `data/` | `approx_weights.bin` (trained net, 64 KB), `test_dataset.csv` (15,000 labelled states), `sample_dataset_preview.csv` (first 500 rows, 6 columns, easy to open), `validation_output.txt`, `approx_eval_output.txt`. |

```
                      hand_eval.hpp  (score 2-7 cards)
                            |
   main.cpp ------------> equity.cpp  (exact / mc / hybrid)
      |  \                      |
      |   \--> features.cpp     +--> labels --> gen-dataset --> train.csv --> train_approx.py
      |            |                                                              |
      |            v                                                              v
      +-------> approx.cpp <------------------ approx_weights.bin <---------------+
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
`main.cpp`: one `cmd_*` function per subcommand; `Args`/`parse_args` read `--key value` options.

## Data: synthetic, and why
No dataset of true equities exists, so labels come from simulating the exact process being modelled (a uniformly
random deal), computed by the engine. Turn and river labels are exact; flop and preflop labels are stratified Monte
Carlo with mean standard error 0.18 points (column `eq_se`), far below the network's error, so it does not limit accuracy.

## Run
```
make              # builds ./poker and ./validate (use a directory that allows executables)
make test         # exhaustive evaluator check + statistical checks (~40 s)
./poker equity --hero AsKs --board Qs7s2d --opp "QQ+,AKs:0.5" --opp random --samples 200000
./poker equity --hero AsKs --board Qs7s2d --method exact
./poker gen-dataset --n 120000 --seed 1 --out train.csv
python python/train_approx.py --train train.csv --test data/test_dataset.csv --hidden 128 64 --epochs 300
./poker approx --weights data/approx_weights.bin --hero AsKs --board Qs7s2d
./poker approx-eval --weights data/approx_weights.bin --data data/test_dataset.csv
```
Range syntax: `AsKs` exact combo, `QQ` pocket pair, `AKs`/`AKo`/`AK` suited/offsuit/both, `TT+`, `ATs+`,
`:w` relative weight, `random` any hand. Use one `--opp` per opponent.
