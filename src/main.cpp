// main.cpp - command-line front end for the engine. Subcommands:
//
//   poker equity --hero AsKs [--board Qs7s2d] [--opp random | --opp "QQ+,AKs:0.5"]...
//                [--method hybrid|mc|exact] [--samples N] [--seed S]
//   poker score  < file         one hand per line (2-7 cards); prints category and score
//   poker gen-dataset --n N --seed S --out FILE      training data for the neural net
//   poker approx --weights W1[,W2,W3,W4] --hero AsKs [--board Qs7s2d]
//                                 neural-net equity vs 1 random opponent
//   poker approx-eval --weights W1[,W2,W3,W4] --data FILE
//                                 accuracy + speed of the net on a dataset
//
// --weights is either one PKAP file (single model) or four comma-separated files
// (per-street models in order: preflop, flop, turn, river).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

#include "approx.hpp"
#include "equity.hpp"
#include "features.hpp"

using namespace pk;

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// --key value pairs; a key may repeat (e.g. several --opp).
struct Args {
  std::multimap<std::string, std::string> kv;
  std::string get(const std::string& k, const std::string& d = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? d : it->second;
  }
  std::vector<std::string> all(const std::string& k) const {
    std::vector<std::string> v;
    auto r = kv.equal_range(k);
    for (auto it = r.first; it != r.second; ++it) v.push_back(it->second);
    return v;
  }
};
static Args parse_args(int argc, char** argv, int start) {
  Args a;
  for (int i = start; i + 1 < argc; i += 2) {
    if (std::strncmp(argv[i], "--", 2) != 0) throw std::invalid_argument(std::string("expected --option, got ") + argv[i]);
    a.kv.emplace(argv[i] + 2, argv[i + 1]);
  }
  return a;
}

static std::vector<Card> board_arg(const std::string& s) {
  return (s.empty() || s == "-") ? std::vector<Card>{} : parse_cards(s);
}

// Split a comma-separated list of file paths. Empty entries are dropped.
static std::vector<std::string> split_paths(const std::string& spec) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : spec) {
    if (c == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); }
    else cur.push_back(c);
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

// Load one PKAP file (single model) or four (per-street, order preflop/flop/turn/river).
static std::vector<Net> load_nets(const std::string& spec) {
  std::vector<std::string> paths = split_paths(spec);
  if (paths.empty()) throw std::runtime_error("--weights is required");
  if (paths.size() != 1 && paths.size() != 4)
    throw std::runtime_error("--weights must be one file or four comma-separated files (per-street)");
  std::vector<Net> nets;
  nets.reserve(paths.size());
  for (const auto& p : paths) nets.push_back(Net::load(p));
  return nets;
}

static int street_of(int nb) {
  return nb == 0 ? 0 : nb == 3 ? 1 : nb == 4 ? 2 : 3;
}

// ---------------------------------------------------------------------------
static int cmd_equity(const Args& a) {
  auto hero = parse_cards(a.get("hero"));
  auto board = board_arg(a.get("board"));
  std::vector<Range> opps;
  for (const std::string& o : a.all("opp")) opps.push_back(parse_range(o));
  if (opps.empty()) opps.push_back(Range{});  // default: one random opponent
  const std::string method = a.get("method", "hybrid");
  const uint64_t samples = std::stoull(a.get("samples", "200000"));
  const uint64_t seed = std::stoull(a.get("seed", "1"));

  double t0 = now();
  EquityResult r = method == "exact" ? exact_equity(hero, board, opps)
                 : method == "mc"    ? mc_equity(hero, board, opps, samples, seed)
                                     : hybrid_equity(hero, board, opps, samples, seed);
  double dt = now() - t0;
  std::printf("method=%s opponents=%zu showdowns=%llu %s time=%.3fs\n", method.c_str(), opps.size(),
              (unsigned long long)r.samples, r.exact ? "(exact)" : "(estimate)", dt);
  std::printf("  win    %7.3f%%  +/- %.3f\n", 100 * r.win.value, 100 * r.win.se);
  std::printf("  tie    %7.3f%%  +/- %.3f\n", 100 * r.tie.value, 100 * r.tie.se);
  std::printf("  equity %7.3f%%  +/- %.3f   (win + pot share of ties)\n", 100 * r.equity.value, 100 * r.equity.se);
  return 0;
}

static int cmd_score() {
  std::string line;
  while (std::getline(std::cin, line)) {
    auto cards = parse_cards(line);
    uint32_t s = evaluate(cards.data(), int(cards.size()));
    std::printf("%u %s\n", s, category_name(s >> 20));
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Training-set generator. Each row: a random (hero, board) state at a random street,
// labelled with P(win), P(tie) vs ONE random opponent.
//   river / turn labels are EXACT (hybrid enumerates every runout),
//   flop / preflop labels are stratified Monte Carlo with a reported standard error.
static int cmd_gen(const Args& a) {
  const long n = std::stol(a.get("n", "1000"));
  const uint64_t seed = std::stoull(a.get("seed", "1"));
  const std::string out = a.get("out", "dataset.csv");
  FILE* fp = std::fopen(out.c_str(), "w");
  if (!fp) throw std::runtime_error("cannot write " + out);
  std::fprintf(fp, "hero,board,street,win,tie,eq_se");
  for (int i = 0; i < kNumFeatures; ++i) std::fprintf(fp, ",%s", feature_name(i));
  std::fprintf(fp, "\n");

  const int board_len[4] = {0, 3, 4, 5};
  const uint64_t per_stratum[4] = {12, 12, 45, 2};  // attempts per opponent holding (45 runouts => turn is exact)
  double t0 = now();
  for (long i = 0; i < n; ++i) {
    std::mt19937_64 rng(seed * 1000003ULL + uint64_t(i));
    const int street = int(rng() % 4);
    const int nb = board_len[street];
    Card deck[52];
    for (int c = 0; c < 52; ++c) deck[c] = Card(c);
    for (int k = 0; k < 2 + nb; ++k) std::swap(deck[k], deck[k + rng() % (52 - k)]);
    Card hero[2] = {deck[0], deck[1]};
    std::vector<Card> board(deck + 2, deck + 2 + nb);
    std::vector<Card> hv(hero, hero + 2);

    const int live = 52 - 2 - nb;
    const uint64_t strata = uint64_t(live) * (live - 1) / 2;
    EquityResult r = hybrid_equity(hv, board, {Range{}}, per_stratum[street] * strata, rng());

    float f[kNumFeatures];
    compute_features(hero, board.data(), nb, f);
    std::string hs = card_str(hero[0]) + card_str(hero[1]), bs;
    for (Card c : board) bs += card_str(c);
    if (bs.empty()) bs = "-";
    std::fprintf(fp, "%s,%s,%d,%.6f,%.6f,%.6f", hs.c_str(), bs.c_str(), street, r.win.value, r.tie.value,
                 r.equity.se);
    for (int k = 0; k < kNumFeatures; ++k) std::fprintf(fp, ",%.6g", f[k]);
    std::fprintf(fp, "\n");
    if ((i + 1) % 20000 == 0) std::fprintf(stderr, "  %ld / %ld rows (%.0fs)\n", i + 1, n, now() - t0);
  }
  std::fclose(fp);
  return 0;
}

// ---------------------------------------------------------------------------
static int cmd_approx(const Args& a) {
  auto hero = parse_cards(a.get("hero"));
  auto board = board_arg(a.get("board"));
  std::vector<Net> nets = load_nets(a.get("weights"));
  const int nb = int(board.size());
  const int street = street_of(nb);

  float f[kNumFeatures], w, t;
  compute_features(hero.data(), board.data(), nb, f);
  const Net& net = (nets.size() == 1) ? nets[0] : nets[street];
  net.predict(f, w, t);
  std::printf("network%s: win %.2f%%  tie %.2f%%  equity %.2f%%  (vs one random opponent)\n",
              nets.size() == 4 ? " (per-street)" : "",
              100 * w, 100 * t, 100 * (w + t / 2));
  return 0;
}

struct Row { std::vector<Card> hero, board; int street; double win, tie, eq_se; };

static int cmd_approx_eval(const Args& a) {
  std::vector<Net> nets = load_nets(a.get("weights"));
  std::ifstream in(a.get("data"));
  if (!in) throw std::runtime_error("cannot open data file");
  std::string line;
  std::getline(in, line);  // header
  std::vector<Row> rows;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string hero, board, st, w, t, se;
    std::getline(ss, hero, ','); std::getline(ss, board, ','); std::getline(ss, st, ',');
    std::getline(ss, w, ','); std::getline(ss, t, ','); std::getline(ss, se, ',');
    rows.push_back({parse_cards(hero), board_arg(board), std::stoi(st), std::stod(w), std::stod(t), std::stod(se)});
  }

  // accuracy + speed of the network (feature computation included in the timing)
  double err[4][3] = {{0}}, cnt[4] = {0};
  std::vector<float> sink(rows.size());
  double t0 = now();
  for (size_t i = 0; i < rows.size(); ++i) {
    float f[kNumFeatures], w, t;
    compute_features(rows[i].hero.data(), rows[i].board.data(), int(rows[i].board.size()), f);
    const Net& net = (nets.size() == 1) ? nets[0] : nets[rows[i].street];
    net.predict(f, w, t);
    sink[i] = w + t / 2;
    const Row& r = rows[i];
    err[r.street][0] += std::fabs(w - r.win);
    err[r.street][1] += std::fabs(t - r.tie);
    err[r.street][2] += std::fabs((w + t / 2) - (r.win + r.tie / 2));
    cnt[r.street] += 1;
  }
  double dt = now() - t0;
  const char* names[4] = {"preflop", "flop", "turn", "river"};
  std::printf("network%s: %zu parameters per model (%.1f KB as float32); %zu rows\n",
              nets.size() == 4 ? " (per-street)" : "",
              nets[0].num_params(), nets[0].num_params() * 4 / 1024.0, rows.size());
  double tot[3] = {0, 0, 0};
  for (int s = 0; s < 4; ++s) {
    std::printf("  %-8s n=%6.0f  MAE win %.2f%%  tie %.2f%%  equity %.2f%%\n", names[s], cnt[s],
                100 * err[s][0] / cnt[s], 100 * err[s][1] / cnt[s], 100 * err[s][2] / cnt[s]);
    for (int k = 0; k < 3; ++k) tot[k] += err[s][k];
  }
  std::printf("  overall  MAE win %.2f%%  tie %.2f%%  equity %.2f%%\n",
              100 * tot[0] / rows.size(), 100 * tot[1] / rows.size(), 100 * tot[2] / rows.size());
  std::printf("  speed: %.2f microseconds per prediction (features + network)\n", 1e6 * dt / rows.size());

  // Reference: what does a 1,000-showdown Monte Carlo estimate cost and how accurate is it?
  const size_t nref = std::min<size_t>(rows.size(), 300);
  double mc_err = 0, t1 = now();
  for (size_t i = 0; i < nref; ++i) {
    const Row& r = rows[i];
    EquityResult e = mc_equity(r.hero, r.board, {Range{}}, 1000, 77 + i);
    mc_err += std::fabs(e.equity.value - (r.win + r.tie / 2));
  }
  double dt1 = now() - t1;
  std::printf("  reference: plain Monte Carlo with 1,000 showdowns: %.1f microseconds, equity MAE %.2f%% (first %zu rows)\n",
              1e6 * dt1 / nref, 100 * mc_err / nref, nref);
  return 0;
}

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      std::fprintf(stderr, "usage: poker <equity|score|gen-dataset|approx|approx-eval> [--options]\n");
      return 2;
    }
    const std::string cmd = argv[1];
    Args a = parse_args(argc, argv, 2);
    if (cmd == "equity") return cmd_equity(a);
    if (cmd == "score") return cmd_score();
    if (cmd == "gen-dataset") return cmd_gen(a);
    if (cmd == "approx") return cmd_approx(a);
    if (cmd == "approx-eval") return cmd_approx_eval(a);
    std::fprintf(stderr, "unknown command: %s\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}