// approx.cpp - see approx.hpp.
#include "approx.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace pk {

Net Net::load(const std::string& path) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open weights file: " + path);
  char magic[4];
  int32_t n_layers = 0;
  if (std::fread(magic, 1, 4, fp) != 4 || std::memcmp(magic, "PKAP", 4) != 0 ||
      std::fread(&n_layers, 4, 1, fp) != 1 || n_layers < 1 || n_layers > 16)
    throw std::runtime_error("not a PKAP weights file");
  Net net;
  for (int l = 0; l < n_layers; ++l) {
    int32_t in = 0, out = 0;
    if (std::fread(&in, 4, 1, fp) != 1 || std::fread(&out, 4, 1, fp) != 1 || in < 1 || out < 1)
      throw std::runtime_error("corrupt weights file");
    Layer L{in, out, std::vector<float>(size_t(in) * out), std::vector<float>(out)};
    if (std::fread(L.W.data(), 4, L.W.size(), fp) != L.W.size() ||
        std::fread(L.b.data(), 4, L.b.size(), fp) != L.b.size())
      throw std::runtime_error("truncated weights file");
    net.layers_.push_back(std::move(L));
  }
  std::fclose(fp);
  return net;
}

size_t Net::num_params() const {
  size_t n = 0;
  for (const Layer& L : layers_) n += L.W.size() + L.b.size();
  return n;
}

void Net::predict(const float* x, float& win, float& tie) const {
  float bufA[256], bufB[256];
  const float* in = x;
  float* out = bufA;
  for (size_t l = 0; l < layers_.size(); ++l) {
    const Layer& L = layers_[l];
    for (int j = 0; j < L.out; ++j) out[j] = L.b[j];
    for (int i = 0; i < L.in; ++i) {
      const float xi = in[i];
      const float* row = &L.W[size_t(i) * L.out];
      for (int j = 0; j < L.out; ++j) out[j] += xi * row[j];
    }
    const bool last = (l + 1 == layers_.size());
    for (int j = 0; j < L.out; ++j) {
      float z = out[j];
      out[j] = last ? 1.0f / (1.0f + std::exp(-z)) : (z > 0 ? z : std::exp(z) - 1.0f);  // ELU
    }
    in = out;
    out = (out == bufA) ? bufB : bufA;
  }
  win = in[0];
  tie = in[1];
}

}  // namespace pk
