// src/poly/nemotron_route.hpp - Nemotron-H's router on the host, header only so a host test can
// check it without an engine, a card or a model.
#pragma once

#include <algorithm>
#include <cmath>

namespace strata::poly {

/// sigmoid scores; selection bias added only to the pick; weights are the plain sigmoids, divided
/// by max(sum, 2^-14) when norm, times scale.  tmp: 2 * n_expert floats.  ids and w: n_used.
inline void nemotron_route(const float* logits, int n_expert, int n_used, const float* bias, bool norm,
                           float scale, int* ids, float* w, float* tmp) {
    float* pick = tmp + n_expert;
    for (int e = 0; e < n_expert; ++e) {
        tmp[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = tmp[e] + (bias ? bias[e] : 0.f);
    }
    float sum = 0.f;
    for (int k = 0; k < n_used; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = tmp[best];
        sum += w[k];
    }
    const float denom = norm ? std::max(sum, 6.103515625e-5f) : 1.f;
    for (int k = 0; k < n_used; ++k) w[k] = w[k] / denom * scale;
}

}  // namespace strata::poly
