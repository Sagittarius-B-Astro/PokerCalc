// equity.hpp - public interface of the equity engine.
//
// "Equity" is the hero's expected share of the pot at showdown:
//   win    = P(hero strictly beats every opponent)
//   tie    = P(hero ties for best with at least one opponent)
//   equity = E[pot share], where a k-way tie for best pays 1/k (heads-up: win + tie/2)
// Every estimate carries a standard error (se); exact results have se == 0.
//
// Three estimators of the same quantity:
//   exact_equity   walks every opponent holding x every remaining board card
//   mc_equity      plain Monte Carlo: sample whole deals at random
//   hybrid_equity  stratified: enumerate opponent #1's holdings, sample the rest
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "hand_eval.hpp"

namespace pk {

// One possible two-card holding with a relative likelihood (weight). A "range" is a
// list of holdings; an EMPTY range means "any two cards" (a random hand).
struct Holding { Card a, b; double w; };
using Range = std::vector<Holding>;

struct Estimate { double value = 0, se = 0; };
struct EquityResult {
  Estimate win, tie, equity;
  uint64_t samples = 0;   // showdowns actually evaluated
  bool exact = false;     // true if no randomness was involved
};

// ---- parsing / printing ---------------------------------------------------
Card parse_card(const std::string& s);                    // "As" -> card index
std::vector<Card> parse_cards(const std::string& s);      // "AsKd7c" -> 3 cards
std::string card_str(Card c);
// Range syntax: comma-separated tokens, each with an optional ":weight".
//   "AsKs" one exact combo   "QQ" all pairs of queens   "AKs" suited   "AKo" offsuit
//   "AK" both   "TT+" pairs TT..AA   "ATs+" ATs,AJs,AQs,AKs   "random" = any hand
Range parse_range(const std::string& spec);

// ---- estimators -----------------------------------------------------------
// opps[i] is opponent i's range (empty = random hand). Cards must all be distinct.
EquityResult exact_equity(const std::vector<Card>& hero, const std::vector<Card>& board,
                          const std::vector<Range>& opps, double max_leaves = 3e9);

EquityResult mc_equity(const std::vector<Card>& hero, const std::vector<Card>& board,
                       const std::vector<Range>& opps, uint64_t samples, uint64_t seed);

// `samples` is the total budget, split evenly across opponent #1's holdings.
EquityResult hybrid_equity(const std::vector<Card>& hero, const std::vector<Card>& board,
                           const std::vector<Range>& opps, uint64_t samples, uint64_t seed);

}  // namespace pk
