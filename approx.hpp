// approx.hpp - inference for the learned approximation of equity vs ONE random opponent.
//
// The network is trained in python/train_approx.py and saved to a small binary file.
// File format (little endian): "PKAP", int32 n_layers, then per layer:
//   int32 in, int32 out, float32 W[in][out] (row-major), float32 b[out].
// Hidden layers use ELU, the output layer uses a sigmoid; outputs are (P(win), P(tie)).
#pragma once
#include <string>
#include <vector>

namespace pk {

class Net {
 public:
  static Net load(const std::string& path);
  // x must have in_dim() floats. Equity = win + tie / 2 (heads-up).
  void predict(const float* x, float& win, float& tie) const;
  int in_dim() const { return layers_.front().in; }
  size_t num_params() const;

 private:
  struct Layer { int in, out; std::vector<float> W, b; };
  std::vector<Layer> layers_;
};

}  // namespace pk
