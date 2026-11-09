// src/poly/nemotron_scale_test.cu - the operator composition Nemotron-H uses for an NVFP4 ssm
// projection with a .scale: the projection's RESULT is scaled, not its weights and not twice.
// Host reference: y = scale * (W x); without the .scale tensor the plain product is left alone.
// One row, as nemotronh.cpp's mamba() calls it: the model runs one token a call, and
// scale_rows reads one factor per row.
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

int main() {
    constexpr int n_in = 4, n_out = 3, rows = 1;
    const float w[n_out * n_in] = {1.f, 2.f, 3.f, 4.f,
                                   0.f, 1.f, 0.f, 1.f,
                                   -1.f, 0.5f, 2.f, -2.f};   // rows of n_in, as gemv_f32 reads them
    const float x[rows * n_in] = {1.f, -1.f, 2.f, 0.f};
    constexpr float scale = 2.5f;

    float want_plain[n_out], want_scaled[n_out];
    for (int o = 0; o < n_out; ++o) {
        float a = 0.f;
        for (int i = 0; i < n_in; ++i) a += w[o * n_in + i] * x[i];
        want_plain[o] = a;
        want_scaled[o] = scale * a;
    }

    float *dw = nullptr, *dx = nullptr, *dy = nullptr, *dy_plain = nullptr, *ds = nullptr;
    if (cudaMalloc(&dw, sizeof(w)) != cudaSuccess || cudaMalloc(&dx, sizeof(x)) != cudaSuccess ||
        cudaMalloc(&dy, sizeof(want_scaled)) != cudaSuccess ||
        cudaMalloc(&dy_plain, sizeof(want_plain)) != cudaSuccess ||
        cudaMalloc(&ds, sizeof(float)) != cudaSuccess)   // one factor for the one row
        return 2;
    cudaMemcpy(dw, w, sizeof(w), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, sizeof(x), cudaMemcpyHostToDevice);
    cudaMemcpy(ds, &scale, sizeof(float), cudaMemcpyHostToDevice);

    // projection first, then the scalar on its result, as llama.cpp's build_lora_mm does with w_s;
    // beside it, the same product without the .scale tensor
    strata::poly::gemv_f32(dw, dx, dy, n_in, n_out, n_out, rows, nullptr);
    strata::poly::scale_rows(dy, n_out, n_out, rows, nullptr, ds, false, nullptr);
    strata::poly::gemv_f32(dw, dx, dy_plain, n_in, n_out, n_out, rows, nullptr);

    float got[n_out], got_plain[n_out];
    cudaMemcpy(got, dy, sizeof(got), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_plain, dy_plain, sizeof(got_plain), cudaMemcpyDeviceToHost);
    cudaFree(ds);
    cudaFree(dy_plain);
    cudaFree(dy);
    cudaFree(dx);
    cudaFree(dw);

    float worst = 0.f, worst_plain = 0.f;
    for (int i = 0; i < n_out; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(got_plain[i])) {
            std::fprintf(stderr, "FAIL: non-finite result at %d\n", i);
            return 1;
        }
        worst = std::max(worst, std::fabs(got[i] - want_scaled[i]));
        worst_plain = std::max(worst_plain, std::fabs(got_plain[i] - want_plain[i]));
    }
    std::printf("projection then scale: worst abs %.3g; absent scale: worst abs %.3g\n", worst, worst_plain);
    if (worst > 1e-6f || worst_plain > 1e-6f) {
        std::fprintf(stderr, "FAIL: result scale differs from scale * (W x) or the plain product changed\n");
        return 1;
    }
    std::puts("nemotron scale: the projection's result is multiplied once by the tensor's F32 scale, "
              "and the plain product is untouched");
    return 0;
}
