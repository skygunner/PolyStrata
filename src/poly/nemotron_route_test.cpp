// src/poly/nemotron_route_test.cpp - Nemotron-H's router on the host: sigmoid, bias on the pick
// only, the 2^-14 floor under the weight sum, the scale, and ties to the lower expert.
#include "nemotron_route.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using strata::poly::nemotron_route;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

int main() {
    float tmp[64];
    int ids[8];
    float w[8];
    auto sigmoid = [](float x) { return 1.f / (1.f + std::exp(-x)); };

    // logits 0..5, top 3: 5, 4, 3; weights are the sigmoids, normalized.
    {
        const float logits[6] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        nemotron_route(logits, 6, 3, nullptr, true, 1.f, ids, w, tmp);
        check(ids[0] == 5 && ids[1] == 4 && ids[2] == 3, "top 3 of 0..5");
        const float sum = sigmoid(5.f) + sigmoid(4.f) + sigmoid(3.f);
        for (int k = 0; k < 3; ++k)
            check(std::fabs(w[k] - sigmoid(logits[ids[k]]) / sum) < 1e-6f, "normalized sigmoid weight");
    }
    // Selection bias moves the pick but not the weight: the weight stays the unbiased sigmoid.
    {
        const float logits[6] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        const float bias[6] = {10.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        nemotron_route(logits, 6, 2, bias, true, 1.f, ids, w, tmp);
        check(ids[0] == 0, "bias picks expert 0");
        check(std::fabs(w[0] - sigmoid(0.f) / (sigmoid(0.f) + sigmoid(5.f))) < 1e-6f,
              "bias does not change the weight");
        check(ids[1] == 5, "second pick is the unbiased best");
    }
    // The floor: tiny weights divide by 2^-14, not by their sum.  This is the literal ggml uses.
    {
        const float logits[4] = {-30.f, -30.f, -30.f, -30.f};
        nemotron_route(logits, 4, 2, nullptr, true, 1.f, ids, w, tmp);
        check(ids[0] == 0 && ids[1] == 1, "ties take the lower expert first");
        check(std::fabs(w[0] - sigmoid(-30.f) / 6.103515625e-5f) < 1e-9f, "weight sum floor");
    }
    // No norm: the weights are the sigmoids themselves.  Scale multiplies them.
    {
        const float logits[3] = {1.f, 2.f, 3.f};
        nemotron_route(logits, 3, 3, nullptr, false, 5.f, ids, w, tmp);
        for (int k = 0; k < 3; ++k) check(std::fabs(w[k] - 5.f * sigmoid(logits[ids[k]])) < 1e-6f, "scale, no norm");
    }
    if (failures == 0) std::puts("nemotron_route: 4 cases match");
    return failures == 0 ? 0 : 1;
}
