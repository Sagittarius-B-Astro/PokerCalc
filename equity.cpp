// equity.cpp - implementation of the equity engine (see equity.hpp for the contract).
#include "equity.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>

namespace pk {

// ===========================================================================
// Parsing / printing
// ===========================================================================
static const char* kRanks = "23456789TJQKA";
static const char* kSuits = "cdhs";

static int rank_index(char ch) {
  const char up = char(std::toupper(static_cast<unsigned char>(ch)));
  for (int i = 0; i < 13; ++i) if (kRanks[i] == up) return i;
  return -1;
}
static int suit_index(char ch) {
  const char lo = char(std::tolower(static_cast<unsigned char>(ch)));
  for (int i = 0; i < 4; ++i) if (kSuits[i] == lo) return i;
  return -1;
}

Card parse_card(const std::string& s) {
  if (s.size() != 2) throw std::invalid_argument("bad card: '" + s + "'");
  int r = rank_index(s[0]), su = suit_index(s[1]);
  if (r < 0 || su < 0) throw std::invalid_argument("bad card: '" + s + "'");
  return Card(r * 4 + su);
}

std::vector<Card> parse_cards(const std::string& s) {
  std::string t;
  for (char ch : s) if (!std::isspace(static_cast<unsigned char>(ch))) t.push_back(ch);
  if (t.size() % 2) throw std::invalid_argument("odd number of characters in cards: " + s);
  std::vector<Card> out;
  for (size_t i = 0; i < t.size(); i += 2) out.push_back(parse_card(t.substr(i, 2)));
  return out;
}

std::string card_str(Card c) {
  return std::string(1, kRanks[rank_of(c)]) + kSuits[suit_of(c)];
}

// Range parsing. Holdings are stored once per unordered pair; a later token for the
// same pair overrides an earlier one.
Range parse_range(const std::string& spec) {
  std::map<std::pair<int, int>, double> table;
  auto put = [&](int c1, int c2, double w) {
    if (c1 > c2) std::swap(c1, c2);
    table[{c1, c2}] = w;
  };
  size_t pos = 0;
  while (pos <= spec.size()) {
    size_t comma = spec.find(',', pos);
    if (comma == std::string::npos) comma = spec.size();
    std::string tok = spec.substr(pos, comma - pos);
    pos = comma + 1;
    tok.erase(std::remove_if(tok.begin(), tok.end(), [](unsigned char ch) { return std::isspace(ch); }),
              tok.end());
    if (tok.empty()) continue;

    double w = 1.0;
    size_t colon = tok.find(':');
    if (colon != std::string::npos) {
      w = std::stod(tok.substr(colon + 1));
      tok = tok.substr(0, colon);
    }
    std::string low = tok;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char ch) { return std::tolower(ch); });
    if (low == "random" || low == "any") return Range{};  // any two cards

    // exact combo such as "AsKs"
    if (tok.size() == 4 && rank_index(tok[0]) >= 0 && suit_index(tok[1]) >= 0 &&
        rank_index(tok[2]) >= 0 && suit_index(tok[3]) >= 0) {
      put(parse_card(tok.substr(0, 2)), parse_card(tok.substr(2, 2)), w);
      continue;
    }

    bool plus = !tok.empty() && tok.back() == '+';
    if (plus) tok.pop_back();
    if (tok.size() < 2 || tok.size() > 3) throw std::invalid_argument("bad range token: " + tok);
    int r1 = rank_index(tok[0]), r2 = rank_index(tok[1]);
    if (r1 < 0 || r2 < 0) throw std::invalid_argument("bad range token: " + tok);
    char kind = tok.size() == 3 ? char(std::tolower(static_cast<unsigned char>(tok[2]))) : ' ';
    if (kind != ' ' && kind != 's' && kind != 'o') throw std::invalid_argument("bad range token: " + tok);

    if (r1 == r2) {  // pocket pairs
      if (kind != ' ') throw std::invalid_argument("pairs cannot be suited/offsuit: " + tok);
      for (int r = r1; r <= (plus ? 12 : r1); ++r)
        for (int s1 = 0; s1 < 4; ++s1)
          for (int s2 = s1 + 1; s2 < 4; ++s2) put(r * 4 + s1, r * 4 + s2, w);
    } else {
      int hi = std::max(r1, r2), lo = std::min(r1, r2);
      for (int l = lo; l <= (plus ? hi - 1 : lo); ++l)
        for (int s1 = 0; s1 < 4; ++s1)
          for (int s2 = 0; s2 < 4; ++s2) {
            bool suited = (s1 == s2);
            if ((suited && kind == 'o') || (!suited && kind == 's')) continue;
            put(hi * 4 + s1, l * 4 + s2, w);
          }
    }
  }
  Range out;
  for (auto& kv : table) out.push_back({Card(kv.first.first), Card(kv.first.second), kv.second});
  return out;
}

// ===========================================================================
// Shared machinery
// ===========================================================================
namespace {

using Rng = std::mt19937_64;

inline uint64_t bit(int c) { return 1ULL << c; }
inline double unit(Rng& r) { return double(r() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)

double binom(int n, int k) {
  if (k < 0 || k > n) return 0.0;
  double r = 1.0;
  for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
  return r;
}

// Call f(chosen) for every k-subset of pool[0..n). `chosen` points at k cards.
template <class F>
void for_each_combo(const Card* pool, int n, int k, F&& f) {
  if (k == 0) { f(pool); return; }
  if (k > n) return;
  int idx[8];
  Card chosen[8];
  for (int i = 0; i < k; ++i) idx[i] = i;
  while (true) {
    for (int i = 0; i < k; ++i) chosen[i] = pool[idx[i]];
    f(chosen);
    int i = k - 1;
    while (i >= 0 && idx[i] == n - k + i) --i;
    if (i < 0) return;
    ++idx[i];
    for (int j = i + 1; j < k; ++j) idx[j] = idx[j - 1] + 1;
  }
}

// What happened at one showdown from the hero's point of view.
struct Outcome { double win, tie, eq; };

// `board5` holds the 5 community cards, `opp` holds 2*n_opp cards.
inline Outcome showdown(const Card hero[2], const Card* board5, const Card* opp, int n_opp) {
  Card c[7];
  for (int i = 0; i < 5; ++i) c[i] = board5[i];
  c[5] = hero[0]; c[6] = hero[1];
  const uint32_t hs = evaluate(c, 7);
  uint32_t best = 0;
  int n_best = 0;  // how many opponents share the best opponent score
  for (int i = 0; i < n_opp; ++i) {
    c[5] = opp[2 * i]; c[6] = opp[2 * i + 1];
    const uint32_t s = evaluate(c, 7);
    if (s > best) { best = s; n_best = 1; }
    else if (s == best) ++n_best;
  }
  if (hs > best) return {1.0, 0.0, 1.0};
  if (hs < best) return {0.0, 0.0, 0.0};
  return {0.0, 1.0, 1.0 / (1 + n_best)};  // (n_best + 1)-way tie for the pot
}

// Everything derived once from the user's inputs.
struct Setup {
  Card hero[2];
  std::vector<Card> board;
  int need = 0;                 // board cards still to come
  uint64_t dead = 0;            // hero + board cards
  std::vector<bool> is_random;  // per opponent
  std::vector<Range> ranges;    // per opponent: holdings not touching dead cards
};

Setup make_setup(const std::vector<Card>& hero, const std::vector<Card>& board,
                 const std::vector<Range>& opps) {
  if (hero.size() != 2) throw std::invalid_argument("hero needs exactly 2 cards");
  if (!(board.empty() || board.size() == 3 || board.size() == 4 || board.size() == 5))
    throw std::invalid_argument("board must have 0, 3, 4 or 5 cards");
  if (opps.empty()) throw std::invalid_argument("need at least one opponent");
  Setup s;
  s.hero[0] = hero[0]; s.hero[1] = hero[1];
  s.board = board;
  s.need = 5 - int(board.size());
  for (Card c : hero) { if (s.dead & bit(c)) throw std::invalid_argument("duplicate card"); s.dead |= bit(c); }
  for (Card c : board) { if (s.dead & bit(c)) throw std::invalid_argument("duplicate card"); s.dead |= bit(c); }
  if (52 - __builtin_popcountll(s.dead) < int(2 * opps.size()) + s.need)
    throw std::invalid_argument("not enough cards left in the deck");
  for (const Range& r : opps) {
    s.is_random.push_back(r.empty());
    Range f;
    for (const Holding& h : r) {
      if (h.w <= 0 || h.a == h.b) continue;
      if ((s.dead & bit(h.a)) || (s.dead & bit(h.b))) continue;  // card removal
      f.push_back(h);
    }
    if (!r.empty() && f.empty()) throw std::invalid_argument("an opponent range has no possible holdings");
    s.ranges.push_back(std::move(f));
  }
  return s;
}

// Weighted random choice from a range.
struct Picker {
  std::vector<Holding> h;
  std::vector<double> cum;
  explicit Picker(const Range& r) : h(r) {
    double t = 0;
    for (const Holding& x : h) { t += x.w; cum.push_back(t); }
  }
  const Holding& pick(Rng& rng) const {
    double u = unit(rng) * cum.back();
    size_t i = size_t(std::upper_bound(cum.begin(), cum.end(), u) - cum.begin());
    return h[std::min(i, h.size() - 1)];
  }
};

// Deals "everything except opponent #first" at random.
//   * ranged opponents: each draws a holding from its own range independently;
//     if ANY card repeats or is dead, the WHOLE combination is rejected (draw returns false).
//     This is exactly a draw from the joint distribution restricted to legal deals.
//   * random opponents and the remaining board cards: partial shuffle of what is left.
class Sampler {
 public:
  Sampler(const Setup& s, size_t first_other, uint64_t seed) : s_(s), rng_(seed) {
    for (size_t i = first_other; i < s.ranges.size(); ++i) {
      if (s.is_random[i]) ++n_random_;
      else pickers_.emplace_back(s.ranges[i]);
    }
  }

  // has_fixed: opponent #0 is held fixed at (fa, fb) (hybrid mode).
  bool draw(bool has_fixed, Card fa, Card fb, Outcome& out) {
    uint64_t used = s_.dead;
    Card opp[24];
    int k = 0;
    if (has_fixed) { opp[k++] = fa; opp[k++] = fb; used |= bit(fa) | bit(fb); }
    for (const Picker& p : pickers_) {
      const Holding& h = p.pick(rng_);
      if ((used & bit(h.a)) || (used & bit(h.b))) return false;  // collision: reject whole deal
      used |= bit(h.a) | bit(h.b);
      opp[k++] = h.a; opp[k++] = h.b;
    }
    Card pool[52];
    int L = 0;
    for (int c = 0; c < 52; ++c) if (!(used & bit(c))) pool[L++] = Card(c);
    const int take = 2 * n_random_ + s_.need;
    for (int i = 0; i < take; ++i) {  // partial Fisher-Yates shuffle
      int j = i + int(rng_() % uint64_t(L - i));
      std::swap(pool[i], pool[j]);
    }
    for (int i = 0; i < 2 * n_random_; ++i) opp[k++] = pool[i];
    Card board5[5];
    int nb = 0;
    for (Card c : s_.board) board5[nb++] = c;
    for (int i = 0; i < s_.need; ++i) board5[nb++] = pool[2 * n_random_ + i];
    out = showdown(s_.hero, board5, opp, (has_fixed ? 1 : 0) + int(pickers_.size()) + n_random_);
    return true;
  }

 private:
  const Setup& s_;
  Rng rng_;
  std::vector<Picker> pickers_;
  int n_random_ = 0;
};

}  // namespace

// ===========================================================================
// Plain Monte Carlo
// ===========================================================================
EquityResult mc_equity(const std::vector<Card>& hero, const std::vector<Card>& board,
                       const std::vector<Range>& opps, uint64_t samples, uint64_t seed) {
  if (samples < 2) throw std::invalid_argument("need at least 2 samples");
  Setup s = make_setup(hero, board, opps);
  Sampler smp(s, 0, seed);
  double sum[3] = {0, 0, 0}, sq[3] = {0, 0, 0};
  uint64_t n = 0, rejected_in_a_row = 0;
  while (n < samples) {
    Outcome o;
    if (!smp.draw(false, 0, 0, o)) {
      if (++rejected_in_a_row > 5000000) throw std::runtime_error("ranges almost never fit together");
      continue;
    }
    rejected_in_a_row = 0;
    const double v[3] = {o.win, o.tie, o.eq};
    for (int q = 0; q < 3; ++q) { sum[q] += v[q]; sq[q] += v[q] * v[q]; }
    ++n;
  }
  EquityResult r;
  Estimate* dst[3] = {&r.win, &r.tie, &r.equity};
  for (int q = 0; q < 3; ++q) {
    double mean = sum[q] / n;
    double var = std::max(0.0, (sq[q] / n - mean * mean) * double(n) / double(n - 1));
    dst[q]->value = mean;
    dst[q]->se = std::sqrt(var / n);
  }
  r.samples = n;
  return r;
}

// ===========================================================================
// Hybrid (stratified): enumerate opponent #1's holdings, sample the rest
// ===========================================================================
// Why this helps: the biggest source of randomness in a heads-up equity estimate is
// WHICH two cards the opponent holds (there are only ~1,000 possibilities, but they
// differ a lot in strength). Visiting every one of them equally often removes that
// source of variance; only the board runout is left to chance.
//
// Maths: let stratum h be "opponent #1 holds h" with prior weight w_h. For each
// attempt i in stratum h let z_i = 1 if the other players' deal was legal (else 0)
// and y_i the hero's outcome (0 if illegal). Then
//     value = sum_h w_h * mean(y) / sum_h w_h * mean(z)         (a ratio estimator)
// With one opponent z is always 1 and this is just the weighted average over holdings.
// The standard error comes from the delta method on the per-attempt quantity y - R*z.
EquityResult hybrid_equity(const std::vector<Card>& hero, const std::vector<Card>& board,
                           const std::vector<Range>& opps, uint64_t samples, uint64_t seed) {
  Setup s = make_setup(hero, board, opps);

  // Strata = opponent #1's possible holdings.
  Range strata;
  if (s.is_random[0]) {
    for (int a = 0; a < 52; ++a)
      for (int b = a + 1; b < 52; ++b)
        if (!(s.dead & (bit(a) | bit(b)))) strata.push_back({Card(a), Card(b), 1.0});
  } else {
    strata = s.ranges[0];
  }
  const uint64_t m = std::max<uint64_t>(2, samples / strata.size());  // attempts per stratum

  const bool heads_up = opps.size() == 1;
  const int remaining_after_opp1 = 52 - __builtin_popcountll(s.dead) - 2;
  const double runouts = binom(remaining_after_opp1, s.need);
  const bool enumerate = heads_up && runouts <= double(m);  // heads-up: do each stratum exactly

  struct Acc { double w, sz = 0, sy[3] = {0, 0, 0}, syy[3] = {0, 0, 0}; double m = 0; bool exact = false; };
  std::vector<Acc> acc(strata.size());
  Sampler smp(s, 1, seed);
  uint64_t evaluated = 0;

  for (size_t h = 0; h < strata.size(); ++h) {
    Acc& a = acc[h];
    a.w = strata[h].w;
    const Card fa = strata[h].a, fb = strata[h].b;
    if (enumerate) {
      uint64_t used = s.dead | bit(fa) | bit(fb);
      Card pool[52];
      int L = 0;
      for (int c = 0; c < 52; ++c) if (!(used & bit(c))) pool[L++] = Card(c);
      Card opp[2] = {fa, fb};
      for_each_combo(pool, L, s.need, [&](const Card* run) {
        Card board5[5];
        int nb = 0;
        for (Card c : s.board) board5[nb++] = c;
        for (int i = 0; i < s.need; ++i) board5[nb++] = run[i];
        Outcome o = showdown(s.hero, board5, opp, 1);
        a.sy[0] += o.win; a.sy[1] += o.tie; a.sy[2] += o.eq;
        a.sz += 1; a.m += 1; ++evaluated;
      });
      a.exact = true;
    } else {
      uint64_t rejected_in_a_row = 0;
      for (uint64_t t = 0; t < m; ++t) {
        a.m += 1;
        Outcome o;
        if (!smp.draw(true, fa, fb, o)) {  // illegal deal for the other players: z = 0, y = 0
          if (++rejected_in_a_row > 5000000) throw std::runtime_error("ranges almost never fit together");
          continue;
        }
        rejected_in_a_row = 0;
        const double v[3] = {o.win, o.tie, o.eq};
        for (int q = 0; q < 3; ++q) { a.sy[q] += v[q]; a.syy[q] += v[q] * v[q]; }
        a.sz += 1; ++evaluated;
      }
    }
  }

  // Combine strata.
  double D = 0, N[3] = {0, 0, 0};
  for (const Acc& a : acc) {
    D += a.w * a.sz / a.m;
    for (int q = 0; q < 3; ++q) N[q] += a.w * a.sy[q] / a.m;
  }
  if (D <= 0) throw std::runtime_error("no legal deals found");

  EquityResult r;
  Estimate* dst[3] = {&r.win, &r.tie, &r.equity};
  for (int q = 0; q < 3; ++q) {
    const double R = N[q] / D;
    double var = 0;
    for (const Acc& a : acc) {
      if (a.exact) continue;  // an exactly enumerated stratum has no sampling error
      const double sd = a.sy[q] - R * a.sz;                              // sum of d_i
      const double sdd = a.syy[q] - 2 * R * a.sy[q] + R * R * a.sz;      // sum of d_i^2
      const double s2 = std::max(0.0, (sdd - sd * sd / a.m) / (a.m - 1));  // sample variance of d
      var += a.w * a.w * s2 / a.m;
    }
    dst[q]->value = R;
    dst[q]->se = std::sqrt(var) / D;
  }
  r.samples = evaluated;
  r.exact = enumerate;
  return r;
}

// ===========================================================================
// Exact enumeration
// ===========================================================================
namespace {
struct ExactCtx {
  explicit ExactCtx(const Setup& st) : s(st) {}
  const Setup& s;
  std::vector<Range> cands;  // per opponent: candidate holdings
  double acc[3] = {0, 0, 0};
  double total_w = 0;
  uint64_t evaluated = 0;
  Card opp[24] = {};

  void rec(size_t i, uint64_t used, double W) {
    if (i == cands.size()) { leaf(used, W); return; }
    for (const Holding& h : cands[i]) {
      if ((used & bit(h.a)) || (used & bit(h.b))) continue;
      opp[2 * i] = h.a; opp[2 * i + 1] = h.b;
      rec(i + 1, used | bit(h.a) | bit(h.b), W * h.w);
    }
  }
  void leaf(uint64_t used, double W) {
    Card pool[52];
    int L = 0;
    for (int c = 0; c < 52; ++c) if (!(used & bit(c))) pool[L++] = Card(c);
    for_each_combo(pool, L, s.need, [&](const Card* run) {
      Card board5[5];
      int nb = 0;
      for (Card c : s.board) board5[nb++] = c;
      for (int j = 0; j < s.need; ++j) board5[nb++] = run[j];
      Outcome o = showdown(s.hero, board5, opp, int(cands.size()));
      acc[0] += W * o.win; acc[1] += W * o.tie; acc[2] += W * o.eq;
      total_w += W;
      ++evaluated;
    });
  }
};
}  // namespace

EquityResult exact_equity(const std::vector<Card>& hero, const std::vector<Card>& board,
                          const std::vector<Range>& opps, double max_leaves) {
  Setup s = make_setup(hero, board, opps);
  ExactCtx ctx(s);
  double leaves = 1.0;
  const int live = 52 - __builtin_popcountll(s.dead);
  for (size_t i = 0; i < opps.size(); ++i) {
    Range c;
    if (s.is_random[i]) {
      for (int a = 0; a < 52; ++a)
        for (int b = a + 1; b < 52; ++b)
          if (!(s.dead & (bit(a) | bit(b)))) c.push_back({Card(a), Card(b), 1.0});
    } else {
      c = s.ranges[i];
    }
    leaves *= double(c.size());
    ctx.cands.push_back(std::move(c));
  }
  leaves *= binom(live - 2 * int(opps.size()), s.need);
  if (leaves > max_leaves) throw std::runtime_error("exact enumeration too large; use hybrid or mc");
  ctx.rec(0, s.dead, 1.0);
  EquityResult r;
  r.win.value = ctx.acc[0] / ctx.total_w;
  r.tie.value = ctx.acc[1] / ctx.total_w;
  r.equity.value = ctx.acc[2] / ctx.total_w;
  r.samples = ctx.evaluated;
  r.exact = true;
  return r;
}

}  // namespace pk
