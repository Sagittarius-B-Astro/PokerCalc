// features.hpp - turns (hero hole cards, board) into a fixed-length numeric vector.
//
// The same function is used to build the training CSV AND at prediction time, so the
// network can never see differently-computed features in training and in use.
// Every feature is cheap (no enumeration of opponent hands) and SUIT-INVARIANT in the
// sense that relabelling suits gives the same vector, because only counts are used.
// All values are scaled to roughly [0, 1].
#pragma once
#include "hand_eval.hpp"

namespace pk {

constexpr int kNumFeatures = 62;

// board may have 0, 3, 4 or 5 cards (nb). Writes kNumFeatures floats to `out`.
void compute_features(const Card hero[2], const Card* board, int nb, float* out);

// Human-readable feature names (for the CSV header), index-aligned with compute_features.
const char* feature_name(int i);

}  // namespace pk
