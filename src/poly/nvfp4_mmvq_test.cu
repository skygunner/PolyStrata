// src/poly/nvfp4_mmvq_test.cu - the dense NVFP4 matvec against a host dequant of the same blocks.
//
//     poly_nvfp4_mmvq_test
//
// Nemotron's ssm projections are NVFP4 (type 40); this checks the kernel added to decode them in place:
// random blocks with UE4M3 scales in the checkpoint's range plus the encodings that decode to zero, x
// quantized to Q8_1 by the engine's own quantizer and read back, and the kernel's products against
// double-precision dots of the dequantized weights times those same Q8_1 values.  A three-column call must
// give each column the bits its one-column call gives.  The shapes are the model's two projections' widths
// and one narrow one that takes the small-K kernel.
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); std::exit(1); }
}

// The blocks to floats the way the engine's loader does it: ggml's own to_float, the layout's authority
// (the codes are split-half, not byte-order).
void dequant(const uint8_t* w, int count, float* out) {
    const auto* traits = ggml_get_type_traits((ggml_type) 40);
    if (!traits || !traits->to_float) { std::fprintf(stderr, "ggml has no to_float for type 40\n"); std::exit(1); }
    traits->to_float(w, out, (int64_t) count);
}

struct Q81 {   // the engine's Q8_1 block as native_quantize_q8_1 writes it: a half2 (d, sum) then 32 int8
    __half2 ds;
    int8_t qs[32];
};

int run_case(int n_in, int n_out) {
    std::mt19937 rng(40 + n_in + n_out);
    auto byte = [&]() { return (uint8_t)(rng() & 0xFF); };
    // weights: every code, and scales across the range with the two zero encodings and negatives in
    std::vector<uint8_t> w((size_t) n_out * (n_in / 64) * 36);
    for (size_t i = 0; i < w.size(); ++i) w[i] = byte();
    for (size_t b = 0; b < w.size() / 36; ++b) {   // most scales small, as the checkpoint's are
        for (int j = 0; j < 4; ++j)
            if ((rng() & 7) != 0) {
                const uint8_t lo = 0x10 + (rng() % 0x50);
                w[b * 36 + j] = (rng() & 15) < 14 ? lo : (uint8_t)(j == 1 ? 0x00 : j == 2 ? 0x7F : (0x80 | lo));
            }
    }
    std::vector<float> x((size_t) n_in);
    for (float& v : x) v = (float)((int)(rng() % 4001) - 2000) / 97.f;

    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st));
    float* d_x = nullptr;
    void* d_xq = nullptr;
    uint8_t* d_w = nullptr;
    float* d_y = nullptr;
    const size_t xq_bytes = strata::kernels::native_q8_1_bytes(n_in);
    ck(cudaMalloc(&d_x, (size_t) 3 * n_in * 4));
    ck(cudaMalloc(&d_xq, xq_bytes));
    ck(cudaMalloc(&d_w, w.size()));
    ck(cudaMalloc(&d_y, (size_t) 3 * n_out * 4));
    ck(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    ck(cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice));
    strata::kernels::native_quantize_q8_1(d_x, d_xq, n_in, 1, st);
    strata::kernels::native_mmvq(40, d_w, d_xq, d_y, n_in, n_out, 1, st);
    ck(cudaStreamSynchronize(st));

    std::vector<Q81> xq(n_in / 32);
    ck(cudaMemcpy(xq.data(), d_xq, xq.size() * sizeof(Q81), cudaMemcpyDeviceToHost));
    std::vector<float> got(n_out), y1(n_out);
    ck(cudaMemcpy(got.data(), d_y, n_out * 4, cudaMemcpyDeviceToHost));
    ck(cudaMemcpy(y1.data(), d_y, n_out * 4, cudaMemcpyDeviceToHost));   // the memcmp base below

    double num = 0.0, den = 0.0;
    std::vector<float> wf((size_t) n_out * n_in);
    dequant(w.data(), n_out * n_in, wf.data());
    for (int r = 0; r < n_out; ++r) {
        double acc = 0.0;
        const float* row = wf.data() + (size_t) r * n_in;
        for (int blk = 0; blk < n_in / 64; ++blk)
            for (int h = 0; h < 2; ++h) {
                const Q81& q = xq[(size_t) blk * 2 + h];
                const double d = (double) __low2float(q.ds);
                for (int i = 0; i < 32; ++i)
                    acc += (double) row[(size_t) blk * 64 + h * 32 + i] * d * (double) q.qs[i];
            }
        num += std::fabs(acc - (double) got[(size_t) r]);
        den += std::fabs(acc);
    }
    const double rel = den > 0 ? num / den : num;

    // three columns of the same x: each one the bits its own call gives
    std::vector<float> y3((size_t) 3 * n_out);
    void* d_xq3 = nullptr;
    ck(cudaMalloc(&d_xq3, 3 * xq_bytes));
    std::vector<float> x3((size_t) 3 * n_in);
    for (int c = 0; c < 3; ++c) std::copy(x.begin(), x.end(), x3.begin() + (size_t) c * n_in);
    ck(cudaMemcpy(d_x, x3.data(), x3.size() * 4, cudaMemcpyHostToDevice));
    strata::kernels::native_quantize_q8_1(d_x, d_xq3, n_in, 3, st);
    strata::kernels::native_mmvq(40, d_w, d_xq3, d_y, n_in, n_out, 3, st);
    ck(cudaStreamSynchronize(st));
    ck(cudaMemcpy(y3.data(), d_y, 3 * n_out * 4, cudaMemcpyDeviceToHost));
    bool cols_ok = true;
    for (int c = 0; c < 3 && cols_ok; ++c) cols_ok = std::memcmp(y1.data(), y3.data() + (size_t) c * n_out, n_out * 4) == 0;
    ck(cudaFree(d_xq3));

    const bool pass = rel < 1e-3 && cols_ok;
    std::printf("nvfp4 [%5d -> %5d]  relative error %.3g, three columns %s%s\n", n_in, n_out, rel,
                cols_ok ? "the same bits" : "OTHER BITS", pass ? "" : "  FAIL");
    for (void* p : {(void*) d_x, d_xq, (void*) d_w, (void*) d_y}) ck(cudaFree(p));
    ck(cudaStreamDestroy(st));
    return pass ? 0 : 1;
}

}  // namespace

int main() {
    if (strata::kernels::native_mmvq_weight_bytes(40, 4096, 96) != (size_t) 96 * (4096 / 64) * 36) {
        std::fprintf(stderr, "native_mmvq_weight_bytes(40) is not 36 bytes a 64-value block\n");
        return 1;
    }
    int bad = 0;
    bad += run_case(4096, 96);    // ssm_in's width: the plain kernel
    bad += run_case(8192, 64);    // ssm_out's width
    bad += run_case(1024, 128);   // 16 blocks a row: the small-K kernel
    std::printf("%s\n", bad ? "FAIL" : "nvfp4 mmvq: the kernel matches the dequantized blocks and the Q8_1 values");
    return bad;
}
