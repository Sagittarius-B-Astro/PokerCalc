// features.cpp - see features.hpp.
//
// Layout of the 62 features:
//   0-3    street one-hot (preflop, flop, turn, river)
//   4-8    hole cards: high rank, low rank, is-pair, is-suited, rank gap
//   9-21   board: how many cards of each rank (0..4, scaled)
//   22-34  hole cards: how many cards of each rank (pair -> 1.0, single -> 0.5)
//   35-37  flush structure: most cards in one suit (hero+board), hero cards in that suit,
//          most cards of one suit on the board alone
//   38-39  straight structure: most distinct ranks inside any 5-rank window (hero+board / board only)
//   40-48  one-hot of hero's best made hand category (hero + board)
//   49-57  one-hot of the board's own best category (board alone; zeros preflop)
//   58     1 if hero's hole cards improve on the board alone (the fix suggested in the reference paper)
//   59-61  leading rank of hero's hand, second rank of hero's hand, leading rank of the board's hand
#include "features.hpp"

#include <algorithm>
#include <cstdio>

namespace pk {

static int best_window(uint32_t rank_mask) {
  uint32_t m = (rank_mask << 1) | (rank_mask >> 12);  // ace also counts low
  int best = 0;
  for (int lo = 0; lo <= 9; ++lo) best = std::max(best, __builtin_popcount((m >> lo) & 31u));
  return best;
}

void compute_features(const Card hero[2], const Card* board, int nb, float* f) {
  std::fill(f, f + kNumFeatures, 0.0f);
  const int street = nb == 0 ? 0 : nb == 3 ? 1 : nb == 4 ? 2 : 3;
  f[street] = 1.0f;

  const int r0 = rank_of(hero[0]), r1 = rank_of(hero[1]);
  const int hi = std::max(r0, r1), lo = std::min(r0, r1);
  f[4] = hi / 12.0f;
  f[5] = lo / 12.0f;
  f[6] = (hi == lo) ? 1.0f : 0.0f;
  f[7] = (suit_of(hero[0]) == suit_of(hero[1])) ? 1.0f : 0.0f;
  f[8] = (hi - lo) / 12.0f;

  int board_rank_cnt[13] = {0};
  int total_suit[4] = {0}, hero_suit[4] = {0}, board_suit[4] = {0};
  uint32_t board_mask = 0, all_mask = 0;
  for (int i = 0; i < nb; ++i) {
    int r = rank_of(board[i]), s = suit_of(board[i]);
    ++board_rank_cnt[r];
    ++total_suit[s]; ++board_suit[s];
    board_mask |= 1u << r;
  }
  all_mask = board_mask;
  for (int i = 0; i < 2; ++i) {
    int s = suit_of(hero[i]);
    ++total_suit[s]; ++hero_suit[s];
    all_mask |= 1u << rank_of(hero[i]);
  }
  for (int r = 0; r < 13; ++r) f[9 + r] = board_rank_cnt[r] / 4.0f;
  f[22 + r0] += 0.5f;
  f[22 + r1] += 0.5f;

  int best_s = 0;
  for (int s = 1; s < 4; ++s)
    if (total_suit[s] > total_suit[best_s] ||
        (total_suit[s] == total_suit[best_s] && hero_suit[s] > hero_suit[best_s]))
      best_s = s;
  f[35] = total_suit[best_s] / 7.0f;
  f[36] = hero_suit[best_s] / 2.0f;
  f[37] = *std::max_element(board_suit, board_suit + 4) / 5.0f;

  f[38] = best_window(all_mask) / 5.0f;
  f[39] = best_window(board_mask) / 5.0f;

  Card all[7];
  int n = 0;
  for (int i = 0; i < nb; ++i) all[n++] = board[i];
  all[n++] = hero[0];
  all[n++] = hero[1];
  const uint32_t sa = evaluate(all, n);
  f[40 + (sa >> 20)] = 1.0f;
  f[59] = ((sa >> 16) & 15) / 12.0f;
  f[60] = ((sa >> 12) & 15) / 12.0f;
  if (nb >= 3) {
    const uint32_t sb = evaluate(board, nb);
    f[49 + (sb >> 20)] = 1.0f;
    f[58] = sa > sb ? 1.0f : 0.0f;
    f[61] = ((sb >> 16) & 15) / 12.0f;
  }
}

const char* feature_name(int i) {
  static char buf[8][16];
  static int k = 0;
  char* b = buf[k++ & 7];
  std::snprintf(b, 16, "f%d", i);
  return b;
}

}  // namespace pk
