// conv4_silu() with and without a bias, against the host sum: Nemotron-H's Mamba-2 convolution has
// one bias a channel, and the no-bias call must stay the sum Qwen3.5's layers have always had.
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

float sigmoid(float v) { return 1.f / (1.f + std::exp(-v)); }

// The same four-tap sum the kernel walks: state oldest first, then this call's inputs.
void host(const std::vector<float>& x, const std::vector<float>& w, const std::vector<float>& bias, int channels,
          int rows, std::vector<float>& out, std::vector<float>& state) {
    out.assign(x.size(), 0.f);
    for (int c = 0; c < channels; ++c) {
        float s0 = state[(size_t) c * 3 + 0], s1 = state[(size_t) c * 3 + 1], s2 = state[(size_t) c * 3 + 2];
        for (int r = 0; r < rows; ++r) {
            const float v = x[(size_t) r * channels + c];
            float y = s0 * w[(size_t) c * 4 + 0] + s1 * w[(size_t) c * 4 + 1] + s2 * w[(size_t) c * 4 + 2] +
                      v * w[(size_t) c * 4 + 3];
            if (!bias.empty()) y += bias[(size_t) c];
            out[(size_t) r * channels + c] = y * sigmoid(y);
            s0 = s1;
            s1 = s2;
            s2 = v;
        }
        state[(size_t) c * 3 + 0] = s0;
        state[(size_t) c * 3 + 1] = s1;
        state[(size_t) c * 3 + 2] = s2;
    }
}

float worst(const std::vector<float>& a, const std::vector<float>& b) {
    float d = 0.f;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

// rows 3 walks a channel's tokens (conv4_silu_kernel), rows 20 takes them together (the rows kernel).
bool run(int channels, int rows) {
    std::vector<float> x((size_t) rows * channels), w((size_t) channels * 4), bias(channels), state((size_t) channels * 3);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 13) - 6) / 7.f;
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int(i % 9) - 4) / 11.f;
    for (int c = 0; c < channels; ++c) bias[(size_t) c] = 0.25f - 0.05f * c;
    for (size_t i = 0; i < state.size(); ++i) state[i] = float(int(i % 5) - 2) / 3.f;

    std::vector<float> want_state = state, want;
    host(x, w, {}, channels, rows, want, want_state);
    std::vector<float> want_b_state = state, want_b;
    host(x, w, bias, channels, rows, want_b, want_b_state);

    float *dx = nullptr, *dw = nullptr, *db = nullptr, *dout = nullptr, *dstate = nullptr;
    const size_t n = x.size();
    if (cudaMalloc(&dx, n * sizeof(float)) != cudaSuccess || cudaMalloc(&dw, w.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&db, bias.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dout, n * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dstate, state.size() * sizeof(float)) != cudaSuccess)
        return false;
    auto setup = [&] {
        cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dstate, state.data(), state.size() * sizeof(float), cudaMemcpyHostToDevice);
    };
    std::vector<float> got(n), got_state(state.size());

    setup();
    strata::poly::conv4_silu(dx, dstate, dw, dout, channels, rows, nullptr);
    cudaMemcpy(got.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_state.data(), dstate, state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const float plain = std::max(worst(got, want), worst(got_state, want_state));
    const std::vector<float> plain_out = got;

    setup();
    strata::poly::conv4_silu(dx, dstate, dw, dout, channels, rows, nullptr, nullptr, 0, 0, db);
    cudaMemcpy(got.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_state.data(), dstate, state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    const float biased = std::max(worst(got, want_b), worst(got_state, want_b_state));

    // The explicit null bias must be the same bytes as leaving the argument out.
    setup();
    strata::poly::conv4_silu(dx, dstate, dw, dout, channels, rows, nullptr, nullptr, 0, 0, nullptr);
    std::vector<float> null_out(n);
    cudaMemcpy(null_out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost);
    const bool same = null_out == plain_out;

    cudaFree(dstate);
    cudaFree(dout);
    cudaFree(db);
    cudaFree(dw);
    cudaFree(dx);
    std::printf("conv4_silu %2d channels, %2d rows: no bias %.3g, bias %.3g\n", channels, rows, plain, biased);
    if (plain > 2e-5f || biased > 2e-5f || !same) return false;
    return true;
}

}  // namespace

int main() {
    const bool a = run(3, 3);
    const bool b = run(5, 20);
    if (a && b) std::puts("conv4_silu: biased and unbiased match the host, null bias is the plain call");
    return a && b ? 0 : 1;
}
