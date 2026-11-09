// src/poly/cpu_experts_relu2_test.cpp - the public CPU expert path with Nemotron-H's geometry: one up
// matrix, relu(x)^2, a down projection narrower than the token width, and a negative gate.  Also checks
// that the GLU path on the same fixture is unchanged when relu2 is false, and that a same-type
// gate/down pair (the official pack) still has its FF width checked against the quant's blocks.
#include "cpu_experts.hpp"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using strata::poly::CpuExpert;
using strata::poly::Workers;
using strata::poly::cpu_expert_types_ok;
using strata::poly::cpu_experts;

static int failures = 0;
static void check_close(float got, float want, const char* what) {
    if (!(std::fabs(got - want) <= 1e-5f)) {
        std::fprintf(stderr, "FAIL: %s: got %.9g, want %.9g\n", what, got, want);
        ++failures;
    }
}

int main() {
    Workers workers(2);
    // x = (1, 2); gate rows (1,0), (0,1); one up row per expert row for the GLU case.
    const std::vector<float> x{1.f, 2.f};
    const std::vector<float> gate{1.f, 0.f, 0.f, 1.f};
    const std::vector<float> up{5.f, 5.f, 5.f, 5.f};
    // down rows for n_out = 2: (2,3) and (1,1), in n_ff-value rows.
    const std::vector<float> down{2.f, 3.f, 1.f, 1.f};
    // gate and up alias for relu2: it must never read the up pointer.
    const CpuExpert relu{reinterpret_cast<const uint8_t*>(gate.data()), reinterpret_cast<const uint8_t*>(gate.data()),
                         reinterpret_cast<const uint8_t*>(down.data()), 1.f};
    float out[2] = {0.f, 0.f};
    cpu_experts(workers, &relu, 1, 1, 0 /* F32 */, 0, 0, 2, 2, x.data(), 1e9f, false, 2, true, out);
    check_close(out[0], 2.f * 1.f + 3.f * 4.f, "relu2 row 0");      // relu2(1,2) = (1,4)
    check_close(out[1], 1.f * 1.f + 1.f * 4.f, "relu2 row 1");

    // The same expert without relu2 is the documented GLU: silu(min(gate, limit)) * clamp(up).
    const CpuExpert glu{reinterpret_cast<const uint8_t*>(gate.data()), reinterpret_cast<const uint8_t*>(up.data()),
                        reinterpret_cast<const uint8_t*>(down.data()), 1.f};
    float glu_out[2] = {0.f, 0.f};
    cpu_experts(workers, &glu, 1, 1, 0, 0, 0, 2, 2, x.data(), 1e9f, false, 2, false, glu_out);
    const auto silu = [](float v) { return v / (1.f + std::exp(-v)); };
    const float u = 5.f * x[0] + 5.f * x[1];   // the up row dotted with x, per hidden row
    check_close(glu_out[0], 2.f * (silu(1.f) * u) + 3.f * (silu(2.f) * u), "GLU row 0");
    check_close(glu_out[1], 1.f * (silu(1.f) * u) + 1.f * (silu(2.f) * u), "GLU row 1");

    // n_out = 1 < n_in = 2, and a negative gate result: relu(x)^2 is 0 there, where x*x would be 1.
    const std::vector<float> neg_gate{-1.f, 0.f, 0.f, 3.f};   // gate x = (-1, 6)
    const std::vector<float> one_row_down{2.f, 3.f};          // n_out = 1 row of n_ff = 2
    const CpuExpert narrow{reinterpret_cast<const uint8_t*>(neg_gate.data()),
                           reinterpret_cast<const uint8_t*>(neg_gate.data()),
                           reinterpret_cast<const uint8_t*>(one_row_down.data()), 1.f};
    constexpr float guard = -123456.f;
    float narrow_out[2] = {0.f, guard};
    cpu_experts(workers, &narrow, 1, 1, 0, 0, 0, 2, 2, x.data(), 1e9f, false, 1, true, narrow_out);
    check_close(narrow_out[0], 2.f * 0.f + 3.f * 36.f, "relu2 narrower output");   // relu2(-1, 6) = (0, 36)
    if (narrow_out[1] != guard) {
        std::fprintf(stderr, "FAIL: n_out = 1 wrote past its row (guard %.9g -> %.9g)\n", guard, narrow_out[1]);
        ++failures;
    }

    // Same type for gate and down (the official IQ2_XXS pack): the FF width is still checked against it.
    const int q4_0 = (int) GGML_TYPE_Q4_0;   // blocks of 32
    std::string err;
    if (!cpu_expert_types_ok(q4_0, q4_0, 64, 64, err)) {
        std::fprintf(stderr, "FAIL: aligned same-type pair rejected: %s\n", err.c_str());
        ++failures;
    }
    err.clear();
    if (cpu_expert_types_ok(q4_0, q4_0, 64, 30, err)) {
        std::fprintf(stderr, "FAIL: same-type gate/down skipped the FF width (30 is not whole 32-blocks)\n");
        ++failures;
    } else if (err.find("30") == std::string::npos) {
        std::fprintf(stderr, "FAIL: FF rejection does not name the width: %s\n", err.c_str());
        ++failures;
    }

    if (failures == 0) std::puts("cpu_experts: relu2 (wide and narrow output), GLU, and same-type FF alignment match");
    return failures == 0 ? 0 : 1;
}
