// validate.cpp - checks that the evaluator and the three equity estimators are correct.
//
// Part A  evaluator: classify ALL C(52,7) = 133,784,560 seven-card hands and compare the
//         number of hands per category with the published table, and count distinct
//         hand strengths (known: 4,824 for 7-card hands).
// Part B  evaluator: hand-written ordering checks (wheel, kickers, flush vs full house...).
// Part C  estimators vs exact enumeration (heads-up, multiway, ranges with card clashes).
// Part D  honesty of the reported standard errors (coverage) and the variance reduction
//         of the hybrid method versus plain Monte Carlo at the same sample budget.
//
// Everything is deterministic (fixed seeds). Exit code is non-zero if any check fails.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "../src/equity.hpp"

using namespace pk;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_fail;
}
static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static uint32_t score(const char* cards) {
  auto v = parse_cards(cards);
  return evaluate(v.data(), int(v.size()));
}

// ---------------------------------------------------------------------------
static void part_a_exhaustive() {
  std::printf("\nPart A: exhaustive check over all C(52,7) seven-card hands\n");
  const uint64_t expected[9] = {23294460ULL, 58627800ULL, 31433400ULL, 6461620ULL, 6180020ULL,
                                4047644ULL,  3473184ULL,  224848ULL,   41584ULL};
  uint64_t count[9] = {0};
  std::vector<bool> seen(size_t(9) << 20, false);
  double t0 = now();
  Card c[7];
  uint64_t total = 0;
  for (int a = 0; a < 46; ++a) { c[0] = Card(a);
  for (int b = a + 1; b < 47; ++b) { c[1] = Card(b);
  for (int d = b + 1; d < 48; ++d) { c[2] = Card(d);
  for (int e = d + 1; e < 49; ++e) { c[3] = Card(e);
  for (int f = e + 1; f < 50; ++f) { c[4] = Card(f);
  for (int g = f + 1; g < 51; ++g) { c[5] = Card(g);
  for (int h = g + 1; h < 52; ++h) { c[6] = Card(h);
    uint32_t s = evaluate(c, 7);
    ++count[s >> 20];
    seen[s] = true;
    ++total;
  }}}}}}}
  double dt = now() - t0;
  size_t distinct = 0;
  for (bool b : seen) distinct += b;
  std::printf("  evaluated %llu hands in %.1fs (%.1f M hands/s)\n", (unsigned long long)total, dt,
              total / dt / 1e6);
  for (int k = 0; k < 9; ++k)
    std::printf("    %-16s %12llu  (expected %llu)\n", category_name(k), (unsigned long long)count[k],
                (unsigned long long)expected[k]);
  bool all = true;
  for (int k = 0; k < 9; ++k) all &= (count[k] == expected[k]);
  check(total == 133784560ULL, "total hands == C(52,7)");
  check(all, "category counts match the published 7-card table");
  std::printf("  distinct hand strengths found: %zu\n", distinct);
  check(distinct == 4824, "distinct 7-card hand strengths == 4,824");
}

// ---------------------------------------------------------------------------
static void part_b_ordering() {
  std::printf("\nPart B: hand-written ordering checks\n");
  check(score("Ah2d3c4s5h9d9c") < score("6h2d3c4s5h9d9c"), "wheel (A-5 straight) loses to a six-high straight");
  check(score("Ah2h3h4h5hKdKc") > score("KsKdKh2c2dAcAd"), "straight flush beats full house");
  check(score("AsAdAh2c2d9c8d") > score("KsKdKh2c2d9c8d"), "aces full beats kings full");
  check(score("AsKs9s5s2s3d4d") > score("KsQs9s5s2s3d4d"), "ace-high flush beats king-high flush (flush outranks the wheel)");
  check(score("2s2d2h2c3d4d5h") > score("AsAdAhKcKd9c8d"), "four of a kind beats full house");
  check(score("AsAdKh9c8d5c3h") > score("AhAcKd9s7d5h3c"), "same pair, same first two kickers: better third kicker wins");
  check(score("AsAdKh9c8d5c3h") == score("AhAcKd9s8h5d3c"), "identical ranks in different suits tie exactly");
  check(score("AsAdKsKd2c3d4h") > score("AsAdKsKd2c2d3h"), "two pair: kicker is the best leftover rank (4 beats 3)");
  check(score("AsAdKsKdJcJd2h") > score("AsAdKsKdTcTd2h"), "three pairs: the third pair's rank serves as kicker (J beats T)");
  check((score("AsAdKsKd2c2d3h") >> 20) == TWO_PAIR, "three pairs is classified as two pair");
  check((score("AsAdAhAcKsKdKh") >> 20) == QUADS, "quads plus trips is classified as quads");
}

// ---------------------------------------------------------------------------
static std::vector<Card> H(const char* s) { return parse_cards(s); }

static void compare(const char* label, const EquityResult& est, const EquityResult& ex) {
  double z = est.equity.se > 0 ? std::fabs(est.equity.value - ex.equity.value) / est.equity.se : 0.0;
  std::printf("  %-34s equity %7.3f%% +/- %.3f  (exact %7.3f%%)  z=%.2f  n=%llu\n", label,
              100 * est.equity.value, 100 * est.equity.se, 100 * ex.equity.value, z,
              (unsigned long long)est.samples);
  char msg[160];
  std::snprintf(msg, sizeof msg, "%s within 4 standard errors of exact", label);
  check(z < 4.0 && std::fabs(est.win.value - ex.win.value) < 0.02 && std::fabs(est.tie.value - ex.tie.value) < 0.02, msg);
}

static void part_c_estimators() {
  std::printf("\nPart C: estimators vs exact enumeration\n");
  std::vector<Range> rnd1 = {Range{}};
  std::vector<Range> rnd2 = {Range{}, Range{}};

  // heads-up, river: hybrid must be EXACT (one runout, enumerated); plain MC statistical
  {
    auto hero = H("AsKs"), board = H("Qs7s2d9h3c");
    auto ex = exact_equity(hero, board, rnd1);
    auto hy = hybrid_equity(hero, board, rnd1, 100000, 1);
    check(hy.exact && std::fabs(hy.equity.value - ex.equity.value) < 1e-12 && hy.equity.se == 0.0,
          "river heads-up: hybrid is exact (se = 0) and equals enumeration");
    compare("river heads-up, plain MC 50k", mc_equity(hero, board, rnd1, 50000, 2), ex);
  }
  // heads-up, turn
  {
    auto hero = H("AsKs"), board = H("Qs7s2d9h");
    auto ex = exact_equity(hero, board, rnd1);
    auto hy = hybrid_equity(hero, board, rnd1, 200000, 3);
    check(hy.exact && std::fabs(hy.equity.value - ex.equity.value) < 1e-12,
          "turn heads-up: hybrid enumerates every runout and equals enumeration");
  }
  // heads-up, flop: exact is ~1.07M showdowns
  {
    auto hero = H("AsKs"), board = H("Qs7s2d");
    double t0 = now();
    auto ex = exact_equity(hero, board, rnd1);
    std::printf("  flop heads-up exact: %llu showdowns in %.2fs\n", (unsigned long long)ex.samples, now() - t0);
    check(ex.samples == 1070190ULL, "flop heads-up has 1,070,190 leaves (1081 holdings x 990 runouts)");
    compare("flop heads-up, hybrid 40k", hybrid_equity(hero, board, rnd1, 40000, 4), ex);
    compare("flop heads-up, plain MC 40k", mc_equity(hero, board, rnd1, 40000, 5), ex);
  }
  // multiway river (2 random opponents): 894,330 leaves
  {
    auto hero = H("AsKs"), board = H("Qs7s2d9h3c");
    auto ex = exact_equity(hero, board, rnd2);
    compare("river, 2 random opps, plain MC 50k", mc_equity(hero, board, rnd2, 50000, 6), ex);
    compare("river, 2 random opps, hybrid 50k", hybrid_equity(hero, board, rnd2, 50000, 7), ex);
  }
  // ranges that fight over the same cards (card removal matters): whole-combination rejection
  {
    auto hero = H("AsKd"), board = H("Qs7s2d9h");
    Range r1 = parse_range("AA,KK,QQ:2,AKs:0.5");   // weights: QQ twice as likely, AKs half
    Range r2 = parse_range("AA,KK,JJ,AQs+");
    std::vector<Range> opps = {r1, r2};
    auto ex = exact_equity(hero, board, opps);
    std::printf("  (range test: exact enumeration has %llu legal showdowns)\n", (unsigned long long)ex.samples);
    // one seeded draw can land in the tail by chance, so judge calibration over many repeats
    double zs[2] = {0, 0}, zz[2] = {0, 0};
    const int reps = 60;
    for (int r = 0; r < reps; ++r) {
      EquityResult e[2] = {mc_equity(hero, board, opps, 40000, 100 + r),
                           hybrid_equity(hero, board, opps, 40000, 900 + r)};
      for (int k = 0; k < 2; ++k) {
        double z = (e[k].equity.value - ex.equity.value) / e[k].equity.se;
        zs[k] += z / reps; zz[k] += z * z / reps;
      }
    }
    const char* nm[2] = {"plain MC", "hybrid"};
    for (int k = 0; k < 2; ++k) {
      std::printf("  overlapping ranges, %-8s over %d repeats: mean z %+.2f, rms z %.2f (ideal 0, 1)\n", nm[k], reps, zs[k], std::sqrt(zz[k]));
      char msg[140];
      std::snprintf(msg, sizeof msg, "overlapping ranges, %s: unbiased and SE calibrated", nm[k]);
      check(std::fabs(zs[k]) < 0.35 && std::sqrt(zz[k]) > 0.75 && std::sqrt(zz[k]) < 1.3, msg);
    }
  }
  // k-way splits: AA vs AA on a board where both play the board -> pot splits
  {
    auto hero = H("2c3d"), board = H("AsKsQsJsTs");   // royal flush on board: everyone ties
    std::vector<Range> opps = {Range{}, Range{}};
    auto r = mc_equity(hero, board, opps, 2000, 10);
    check(std::fabs(r.equity.value - 1.0 / 3.0) < 1e-12 && r.tie.value == 1.0 && r.win.value == 0.0,
          "3-way tie on a royal-flush board: win 0, tie 1, equity exactly 1/3");
  }
}

// ---------------------------------------------------------------------------
static void part_d_one_spot(const char* hero_s, const char* board_s) {
  auto hero = H(hero_s), board = H(board_s);
  std::vector<Range> rnd1 = {Range{}};
  const double truth = exact_equity(hero, board, rnd1).equity.value;
  const int reps = 200;
  // 4 attempts per opponent holding; give plain MC exactly the number of showdowns hybrid used
  const uint64_t hybrid_budget = 4 * 1081;
  const uint64_t used = hybrid_equity(hero, board, rnd1, hybrid_budget, 1).samples;

  struct Row { const char* name; int covered = 0; double mean_se = 0, sum = 0, sumsq = 0; };
  Row rows[2] = {{"plain MC"}, {"hybrid"}};
  for (int r = 0; r < reps; ++r) {
    EquityResult e[2] = {mc_equity(hero, board, rnd1, used, 1000 + r),
                         hybrid_equity(hero, board, rnd1, hybrid_budget, 5000 + r)};
    for (int k = 0; k < 2; ++k) {
      double v = e[k].equity.value;
      rows[k].covered += std::fabs(v - truth) <= 1.96 * e[k].equity.se;
      rows[k].mean_se += e[k].equity.se / reps;
      rows[k].sum += v; rows[k].sumsq += v * v;
    }
  }
  double emp[2];
  std::printf("  spot %s | %s heads-up vs random, %llu showdowns per estimate\n", hero_s, board_s,
              (unsigned long long)used);
  for (int k = 0; k < 2; ++k) {
    double mean = rows[k].sum / reps;
    emp[k] = std::sqrt((rows[k].sumsq / reps - mean * mean) * reps / (reps - 1));
    std::printf("    %-8s reported SE %.4f%% | actual spread %.4f%% | 95%% interval covered truth %d/%d\n",
                rows[k].name, 100 * rows[k].mean_se, 100 * emp[k], rows[k].covered, reps);
    char msg[140];
    std::snprintf(msg, sizeof msg, "%s %s: reported SE matches actual spread (within 20%%)", hero_s, rows[k].name);
    check(std::fabs(rows[k].mean_se / emp[k] - 1.0) < 0.20, msg);
    std::snprintf(msg, sizeof msg, "%s %s: 95%% intervals cover truth 90-99%% of the time", hero_s, rows[k].name);
    check(rows[k].covered >= int(0.90 * reps) && rows[k].covered <= int(0.99 * reps) + 1, msg);
  }
  std::printf("    actual spread plain/hybrid = %.2fx (>1 means hybrid is better)\n", emp[0] / emp[1]);
  char msg[140];
  std::snprintf(msg, sizeof msg, "%s: hybrid is no worse than plain MC at equal showdowns (within 5%%)", hero_s);
  check(emp[1] < 1.05 * emp[0], msg);
}

static void part_d_standard_errors() {
  std::printf("\nPart D: honesty of reported standard errors, and variance reduction (200 repeats each)\n");
  part_d_one_spot("AsKs", "Qs7s2d");
  part_d_one_spot("9c9d", "Ah7s2d");
}

int main() {
  part_a_exhaustive();
  part_b_ordering();
  part_c_estimators();
  part_d_standard_errors();
  std::printf("\n%s (%d failed checks)\n", g_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED", g_fail);
  return g_fail ? 1 : 0;
}
