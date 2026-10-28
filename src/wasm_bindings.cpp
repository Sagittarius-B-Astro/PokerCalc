// wasm_bindings.cpp — Emscripten bindings for the equity engine.
// Only compiled for WASM (`make wasm`); the native build ignores it.
#include <emscripten/bind.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "approx.hpp"
#include "equity.hpp"
#include "features.hpp"

using namespace emscripten;
using namespace pk;

namespace {

struct Est        { double value, se; };
struct EquityOut  { Est win, tie, equity; unsigned long long samples; bool exact; };
struct ApproxOut  { double win, tie, equity; };

EquityOut wrap(const EquityResult& r) {
  return { {r.win.value, r.win.se}, {r.tie.value, r.tie.se},
           {r.equity.value, r.equity.se}, r.samples, r.exact };
}

// Ranges arrive joined with '|' (individual ranges use ',' and ':' internally).
std::vector<Range> split_opps(const std::string& s) {
  std::vector<Range> out;
  if (s.empty()) return out;
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t p = s.find('|', pos);
    if (p == std::string::npos) p = s.size();
    out.push_back(parse_range(s.substr(pos, p - pos)));
    pos = p + 1;
  }
  return out;
}

EquityOut js_exact (std::string hero, std::string board, std::string opps, double maxLeaves) {
  return wrap(exact_equity (parse_cards(hero), parse_cards(board), split_opps(opps), maxLeaves));
}
EquityOut js_mc    (std::string hero, std::string board, std::string opps,
                    unsigned long long s, unsigned long long seed) {
  return wrap(mc_equity    (parse_cards(hero), parse_cards(board), split_opps(opps), s, seed));
}
EquityOut js_hybrid(std::string hero, std::string board, std::string opps,
                    unsigned long long s, unsigned long long seed) {
  return wrap(hybrid_equity(parse_cards(hero), parse_cards(board), split_opps(opps), s, seed));
}

// Cache the loaded nets so we don't re-parse the weight file on every call.
ApproxOut js_approx(std::string weights_csv, std::string hero, std::string board) {
  static std::unordered_map<std::string, std::vector<Net>> cache;
  auto it = cache.find(weights_csv);
  if (it == cache.end()) {
    std::vector<Net> nets;
    size_t pos = 0;
    while (pos <= weights_csv.size()) {
      size_t p = weights_csv.find(',', pos);
      if (p == std::string::npos) p = weights_csv.size();
      if (p > pos) nets.push_back(Net::load(weights_csv.substr(pos, p - pos)));
      pos = p + 1;
    }
    if (nets.empty()) throw std::runtime_error("no weights file given");
    it = cache.emplace(weights_csv, std::move(nets)).first;
  }
  const auto& nets = it->second;

  auto h = parse_cards(hero);
  auto b = parse_cards(board);
  const int nb = int(b.size());
  const int street = nb == 0 ? 0 : nb == 3 ? 1 : nb == 4 ? 2 : 3;

  float f[kNumFeatures], w, t;
  compute_features(h.data(), b.data(), nb, f);
  const Net& net = (nets.size() == 1) ? nets[0] : nets[street];
  net.predict(f, w, t);
  return { w, t, w + 0.5 * t };
}

}  // namespace

EMSCRIPTEN_BINDINGS(poker) {
  value_object<Est>("Est")
    .field("value", &Est::value).field("se", &Est::se);
  value_object<EquityOut>("EquityOut")
    .field("win", &EquityOut::win).field("tie", &EquityOut::tie)
    .field("equity", &EquityOut::equity).field("samples", &EquityOut::samples)
    .field("exact", &EquityOut::exact);
  value_object<ApproxOut>("ApproxOut")
    .field("win", &ApproxOut::win).field("tie", &ApproxOut::tie)
    .field("equity", &ApproxOut::equity);
  function("exactEquity",  &js_exact);
  function("mcEquity",     &js_mc);
  function("hybridEquity", &js_hybrid);
  function("approx",       &js_approx);
}