// hand_eval.hpp - a from-scratch Texas Hold'em hand evaluator (no third-party code).
//
// A "score" is a 24-bit integer: HIGHER = BETTER HAND, EQUAL = EXACT TIE.
//   bits 20..23 : hand category (0 = high card ... 8 = straight flush)
//   bits  0..19 : up to five 4-bit rank fields that break ties inside a category,
//                 most important first (e.g. pair rank, then its three kickers).
// Cards are integers 0..51: rank = card >> 2 (0 = Two ... 12 = Ace), suit = card & 3.
//
// This evaluator counts ranks and suits with small arrays/bitmasks (no big lookup
// table). It is slower than the table-driven evaluators discussed in the write-up,
// but needs only an 8 KB table and is easy to verify exhaustively
// (tests/validate.cpp checks it against all C(52,7) = 133,784,560 hands).
#pragma once
#include <array>
#include <cstdint>

namespace pk {

using Card = uint8_t;
constexpr int rank_of(int c) { return c >> 2; }
constexpr int suit_of(int c) { return c & 3; }

enum Category : uint32_t {
  HIGH_CARD = 0, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH
};
constexpr int kNumCategories = 9;

inline const char* category_name(uint32_t cat) {
  static const char* names[] = {"High Card", "Pair", "Two Pair", "Three of a Kind", "Straight",
                                "Flush", "Full House", "Four of a Kind", "Straight Flush"};
  return names[cat];
}

namespace detail {
// For every 13-bit rank mask: the top rank of the best 5-in-a-row straight, or -1.
// The ace also counts as "low", so A-2-3-4-5 (the wheel) is a straight topped by the Five.
inline std::array<int8_t, 8192> make_straight_table() {
  std::array<int8_t, 8192> t{};
  for (uint32_t mask = 0; mask < 8192; ++mask) {
    uint32_t m = (mask << 1) | (mask >> 12);  // bit 0 = ace-low, bit i+1 = rank i
    int8_t best = -1;
    for (int hi = 13; hi >= 4; --hi)            // window covers bits hi-4 .. hi
      if (((m >> (hi - 4)) & 31u) == 31u) { best = int8_t(hi - 1); break; }
    t[mask] = best;
  }
  return t;
}
inline const std::array<int8_t, 8192> kStraightTable = make_straight_table();

// Pack the n highest set ranks of `mask` into n 4-bit fields (highest rank first).
inline uint32_t pack_top(uint32_t mask, int n) {
  uint32_t out = 0;
  int got = 0;
  for (int r = 12; r >= 0 && got < n; --r)
    if ((mask >> r) & 1u) { out = (out << 4) | uint32_t(r); ++got; }
  for (; got < n; ++got) out <<= 4;  // fewer than n ranks present: pad with zeros
  return out;
}
}  // namespace detail

// Score the best 5-card hand contained in `n` cards (n may be 2..7).
inline uint32_t evaluate(const Card* cards, int n) {
  uint32_t suit_mask[4] = {0, 0, 0, 0};  // which ranks are present in each suit
  int cnt[13] = {0};                     // how many cards of each rank
  uint32_t all = 0;                      // which ranks are present at all
  for (int i = 0; i < n; ++i) {
    int r = rank_of(cards[i]);
    suit_mask[suit_of(cards[i])] |= 1u << r;
    ++cnt[r];
    all |= 1u << r;
  }

  // With <= 7 cards at most one suit can hold 5+ cards.
  uint32_t flush_mask = 0;
  for (int s = 0; s < 4; ++s)
    if (__builtin_popcount(suit_mask[s]) >= 5) { flush_mask = suit_mask[s]; break; }
  if (flush_mask) {
    int sh = detail::kStraightTable[flush_mask];
    if (sh >= 0) return (STRAIGHT_FLUSH << 20) | (uint32_t(sh) << 16);
  }

  // Scan ranks high -> low, remembering the best groups of equal ranks.
  int quad = -1, trip1 = -1, trip2 = -1, pair1 = -1, pair2 = -1;
  for (int r = 12; r >= 0; --r) {
    int c = cnt[r];
    if (c == 4) { if (quad < 0) quad = r; }
    else if (c == 3) { if (trip1 < 0) trip1 = r; else if (trip2 < 0) trip2 = r; }
    else if (c == 2) { if (pair1 < 0) pair1 = r; else if (pair2 < 0) pair2 = r; }
  }

  if (quad >= 0) {
    uint32_t rest = all & ~(1u << quad);
    return (QUADS << 20) | (uint32_t(quad) << 16) | (detail::pack_top(rest, 1) << 12);
  }
  if (trip1 >= 0) {
    int p = trip2 > pair1 ? trip2 : pair1;  // the "pair" half of a full house
    if (p >= 0) return (FULL_HOUSE << 20) | (uint32_t(trip1) << 16) | (uint32_t(p) << 12);
  }
  if (flush_mask) return (FLUSH << 20) | detail::pack_top(flush_mask, 5);
  int sh = detail::kStraightTable[all];
  if (sh >= 0) return (STRAIGHT << 20) | (uint32_t(sh) << 16);
  if (trip1 >= 0) {
    uint32_t rest = all & ~(1u << trip1);
    return (TRIPS << 20) | (uint32_t(trip1) << 16) | (detail::pack_top(rest, 2) << 8);
  }
  if (pair2 >= 0) {  // with three pairs among 7 cards the third pair acts as a kicker
    uint32_t rest = all & ~((1u << pair1) | (1u << pair2));
    return (TWO_PAIR << 20) | (uint32_t(pair1) << 16) | (uint32_t(pair2) << 12) |
           (detail::pack_top(rest, 1) << 8);
  }
  if (pair1 >= 0) {
    uint32_t rest = all & ~(1u << pair1);
    return (PAIR << 20) | (uint32_t(pair1) << 16) | (detail::pack_top(rest, 3) << 4);
  }
  return (HIGH_CARD << 20) | detail::pack_top(all, 5);
}

}  // namespace pk
