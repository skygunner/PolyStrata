// Nemotron-H's Mamba-2 scan on the GPU against the direct CPU recurrence, and a prompt row at a
// time against the whole prompt: the final state must not depend on how the rows were chunked.
// Large dt (above the kernel's 20 clamp) is included because that path is not linear in dt.
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

struct Case {
    std::vector<float> x, dt, a, b, c, d, z, norm, state;
    int inner = 0, heads = 0, groups = 0, state_dim = 0;
};

float softplus(float v) { return v > 20.f ? v : std::log1p(std::exp(v)); }

// The reference the kernels should match: rows in order, then the z gate and the grouped RMS norm.
void host(const Case& k, int rows, std::vector<float>& state, std::vector<float>& out) {
    const int head_width = k.inner / k.heads, group_width = k.inner / k.groups;
    out.assign((size_t) rows * k.inner, 0.f);
    for (int row = 0; row < rows; ++row) {
        for (int ch = 0; ch < k.inner; ++ch) {
            const int h = ch / head_width, g = h / (k.heads / k.groups);
            const float delta = softplus(k.dt[(size_t) row * k.heads + h]);
            const float decay = std::exp(delta * k.a[h]);
            float sum = 0.f;
            for (int s = 0; s < k.state_dim; ++s) {
                const size_t si = (size_t) ch * k.state_dim + s;
                const size_t bi = ((size_t) row * k.groups + g) * k.state_dim + s;
                state[si] = state[si] * decay + k.b[bi] * (k.x[(size_t) row * k.inner + ch] * delta);
                sum += state[si] * k.c[bi];
            }
            const float zv = k.z[(size_t) row * k.inner + ch];
            const float silu = zv / (1.f + std::exp(-zv));
            out[(size_t) row * k.inner + ch] = (sum + k.x[(size_t) row * k.inner + ch] * k.d[h]) * silu;
        }
        for (int g = 0; g < k.groups; ++g) {
            float ss = 0.f;
            for (int i = 0; i < group_width; ++i) {
                const float v = out[(size_t) row * k.inner + g * group_width + i];
                ss += v * v;
            }
            const float scale = 1.f / std::sqrt(ss / group_width + 1e-5f);
            for (int i = 0; i < group_width; ++i)
                out[(size_t) row * k.inner + g * group_width + i] *= scale * k.norm[g * group_width + i];
        }
    }
}

float worst(const std::vector<float>& a, const std::vector<float>& b) {
    float w = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;   // NaN/Inf never passes
        w = std::max(w, std::abs(a[i] - b[i]));
    }
    return w;
}

// One call of `rows`, and `rows` calls of one, into the same device buffers.
bool gpu_run(const Case& k, int rows, std::vector<float>& got, std::vector<float>& got_state) {
    float* base = nullptr;
    const size_t total = k.x.size() + k.dt.size() + k.a.size() + k.b.size() + k.c.size() + k.d.size() +
                         k.z.size() + k.norm.size() + k.state.size() + k.x.size() + rows * (size_t) k.inner;
    if (cudaMalloc(&base, total * sizeof(float)) != cudaSuccess) return false;
    size_t at = 0;
    auto put = [&](const std::vector<float>& v) {
        float* p = base + at;
        if (cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return false;
        at += v.size();
        return true;
    };
    if (!(put(k.x) && put(k.dt) && put(k.a) && put(k.b) && put(k.c) && put(k.d) && put(k.z) && put(k.norm) &&
          put(k.state)))
        return false;
    float* go = base + at;
    // The offsets are fixed per row; mamba2_step takes pointers, so each call starts at row r.
    const size_t x_off = 0, dt_off = k.x.size(), a_off = dt_off + k.dt.size(), b_off = a_off + k.a.size(),
                 c_off = b_off + k.b.size(), d_off = c_off + k.c.size(), z_off = d_off + k.d.size(),
                 n_off = z_off + k.z.size(), s_off = n_off + k.norm.size();
    strata::poly::mamba2_step(base + x_off, base + dt_off, base + a_off, base + b_off, base + c_off, base + d_off,
                              base + z_off, base + n_off, base + s_off, go, k.inner, k.heads, k.groups,
                              k.state_dim, rows, 1e-5f, nullptr);
    std::vector<float> out((size_t) rows * k.inner), state(k.state.size());
    if (cudaMemcpy(out.data(), go, out.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (cudaMemcpy(state.data(), base + s_off, state.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    cudaFree(base);
    got = std::move(out);
    got_state = std::move(state);
    return true;
}

bool gpu_run_one_at_a_time(const Case& k, int rows, std::vector<float>& got, std::vector<float>& got_state) {
    float* base = nullptr;
    const size_t total = k.x.size() + k.dt.size() + k.a.size() + k.b.size() + k.c.size() + k.d.size() +
                         k.z.size() + k.norm.size() + k.state.size() + (size_t) rows * k.inner;
    if (cudaMalloc(&base, total * sizeof(float)) != cudaSuccess) return false;
    size_t at = 0;
    auto put = [&](const std::vector<float>& v) {
        float* p = base + at;
        cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
        at += v.size();
    };
    put(k.x); put(k.dt); put(k.a); put(k.b); put(k.c); put(k.d); put(k.z); put(k.norm); put(k.state);
    const size_t dt_off = k.x.size(), a_off = dt_off + k.dt.size(), b_off = a_off + k.a.size(),
                 c_off = b_off + k.b.size(), d_off = c_off + k.c.size(), z_off = d_off + k.d.size(),
                 n_off = z_off + k.z.size(), s_off = n_off + k.norm.size(), out_off = at;
    for (int r = 0; r < rows; ++r) {
        strata::poly::mamba2_step(base + (size_t) r * k.inner, base + dt_off + (size_t) r * k.heads, base + a_off,
                                  base + b_off + (size_t) r * k.groups * k.state_dim,
                                  base + c_off + (size_t) r * k.groups * k.state_dim, base + d_off,
                                  base + z_off + (size_t) r * k.inner, base + n_off, base + s_off,
                                  base + out_off + (size_t) r * k.inner, k.inner, k.heads, k.groups, k.state_dim, 1,
                                  1e-5f, nullptr);
    }
    std::vector<float> out((size_t) rows * k.inner), state(k.state.size());
    cudaMemcpy(out.data(), base + out_off, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(state.data(), base + s_off, state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(base);
    got = std::move(out);
    got_state = std::move(state);
    return true;
}

Case make(int rows, float dt_lo, float dt_hi, int inner, int heads, int groups) {
    Case k;
    k.inner = inner; k.heads = heads; k.groups = groups; k.state_dim = 128;
    k.x.resize((size_t) rows * k.inner);
    k.dt.resize((size_t) rows * k.heads);
    k.z.resize(k.x.size());
    k.b.resize((size_t) rows * k.groups * k.state_dim);
    k.c.resize(k.b.size());
    k.state.resize((size_t) k.inner * k.state_dim);
    k.a.resize((size_t) heads);
    k.d.resize((size_t) heads);
    for (int i = 0; i < heads; ++i) {
        k.a[i] = (i % 2 == 0) ? -0.25f : -0.5f;
        k.d[i] = (i % 2 == 0) ? 0.5f : -0.25f;
    }
    k.norm.resize(k.inner);
    for (size_t i = 0; i < k.x.size(); ++i) k.x[i] = float(int(i % 11) - 5) / 9.f;
    for (size_t i = 0; i < k.z.size(); ++i) k.z[i] = float(int(i % 7) - 3) / 5.f;
    for (size_t i = 0; i < k.dt.size(); ++i)
        k.dt[i] = dt_lo + (dt_hi - dt_lo) * float(i % 3) / 2.f;
    for (size_t i = 0; i < k.b.size(); ++i) {
        k.b[i] = float(int(i % 13) - 6) / 23.f;
        k.c[i] = float(int(i % 17) - 8) / 29.f;
    }
    for (size_t i = 0; i < k.state.size(); ++i) k.state[i] = float(int(i % 19) - 9) / 37.f;
    for (int i = 0; i < k.inner; ++i) k.norm[i] = 0.75f + 0.03f * float(i % 13);   // bounded, repeating
    return k;
}

}  // namespace

int main() {
    int bad = 0;
    struct Run {
        const char* name;
        int rows;
        float lo, hi;
        float tol;
        int inner, heads, groups;
    } runs[] = {{"3 rows, dt -0.5..0.5", 3, -0.5f, 0.5f, 2e-5f, 8, 2, 2},
                {"3 rows, dt 25..40 (past the 20 clamp)", 3, 25.f, 40.f, 2e-4f, 8, 2, 2},
                {"1 row, dt -0.5..0.5", 1, -0.5f, 0.5f, 2e-5f, 8, 2, 2},
                {"3 rows, inner 8192 / heads 128 / groups 8", 3, -0.5f, 0.5f, 1e-4f, 8192, 128, 8}};
    for (const Run& r : runs) {
        const Case k = make(r.rows, r.lo, r.hi, r.inner, r.heads, r.groups);
        std::vector<float> want_state = k.state, want;
        host(k, r.rows, want_state, want);
        std::vector<float> got, got_state;
        if (!gpu_run(k, r.rows, got, got_state)) {
            std::fprintf(stderr, "FAIL: %s: CUDA allocation or copy\n", r.name);
            return 2;
        }
        const float e = std::max(worst(got, want), worst(got_state, want_state));
        std::printf("%-42s one call: worst abs %.3g\n", r.name, e);
        if (e > r.tol) ++bad;
        std::vector<float> chunk, chunk_state;
        if (!gpu_run_one_at_a_time(k, r.rows, chunk, chunk_state)) {
            std::fprintf(stderr, "FAIL: %s: CUDA allocation or copy (row at a time)\n", r.name);
            return 2;
        }
        const float c = std::max(worst(chunk, got), worst(chunk_state, got_state));
        std::printf("%-42s row at a time vs one call: %.3g\n", r.name, c);
        if (c > r.tol) ++bad;
    }

    {   // gather_rows: a prompt batch's ssm_in parts out of an interleaved projection, value for value
        const int rows = 37, inner = 8192, xbc = 256, heads = 128, stride = inner + xbc + heads;
        std::vector<float> src((size_t) rows * stride), want((size_t) rows * inner);
        for (size_t i = 0; i < src.size(); ++i) src[i] = float(int(i % 23) - 11) / 31.f;
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < inner; ++c) want[(size_t) r * inner + c] = src[(size_t) r * stride + c];
        float *d_src = nullptr, *d_dst = nullptr;
        if (cudaMalloc(&d_src, src.size() * 4) != cudaSuccess ||
            cudaMalloc(&d_dst, want.size() * 4) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: gather_rows: CUDA allocation\n");
            return 2;
        }
        cudaMemcpy(d_src, src.data(), src.size() * 4, cudaMemcpyHostToDevice);
        strata::poly::gather_rows(d_src, d_dst, inner, rows, stride, nullptr);
        std::vector<float> got(want.size());
        cudaMemcpy(got.data(), d_dst, want.size() * 4, cudaMemcpyDeviceToHost);
        cudaFree(d_src);
        cudaFree(d_dst);
        const float e = worst(got, want);
        std::printf("%-42s worst abs %.3g\n", "gather_rows 37 rows", e);
        if (e != 0.f) ++bad;   // a copy: it is right when it is exact
    }
    {   // conv4_silu: a batch of rows through the state, one call, against one row a call: the same sums and
        // the same final state.  A batch that skipped the state update (or mixed the rows' history) shows here.
        const int channels = 64, rows = 32;
        std::vector<float> x((size_t) rows * channels), w(4 * channels), bias(channels), state(3 * channels),
            out((size_t) rows * channels), out_want((size_t) rows * channels), state_after(3 * channels),
            state_want(3 * channels);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 13) - 6) / 17.f;
        for (int c = 0; c < channels; ++c)
            for (int t = 0; t < 4; ++t) w[(size_t) t * channels + c] = float(int((t + c) % 5) - 2) / 9.f;
        for (int c = 0; c < channels; ++c) bias[c] = float(int(c % 7) - 3) / 11.f;
        for (int c = 0; c < 3 * channels; ++c) state[c] = float(int(c % 19) - 9) / 37.f;
        float *d_x = nullptr, *d_w = nullptr, *d_b = nullptr, *d_s = nullptr, *d_o = nullptr;
        if (cudaMalloc(&d_x, x.size() * 4) != cudaSuccess || cudaMalloc(&d_w, w.size() * 4) != cudaSuccess ||
            cudaMalloc(&d_b, bias.size() * 4) != cudaSuccess || cudaMalloc(&d_s, state.size() * 4) != cudaSuccess ||
            cudaMalloc(&d_o, out.size() * 4) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: conv4_silu: CUDA allocation\n");
            return 2;
        }
        cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, bias.data(), bias.size() * 4, cudaMemcpyHostToDevice);
        // Reference: one row a call, the state advancing row by row.
        cudaMemcpy(d_s, state.data(), state.size() * 4, cudaMemcpyHostToDevice);
        for (int r = 0; r < rows; ++r)
            strata::poly::conv4_silu(d_x + (size_t) r * channels, d_s, d_w, d_o + (size_t) r * channels, channels, 1,
                                     nullptr, nullptr, 0, 0, d_b);
        cudaMemcpy(out_want.data(), d_o, out_want.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(state_want.data(), d_s, state_want.size() * 4, cudaMemcpyDeviceToHost);
        // The batch, from the same state.
        cudaMemcpy(d_s, state.data(), state.size() * 4, cudaMemcpyHostToDevice);
        strata::poly::conv4_silu(d_x, d_s, d_w, d_o, channels, rows, nullptr, nullptr, 0, 0, d_b);
        cudaMemcpy(out.data(), d_o, out.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(state_after.data(), d_s, state_after.size() * 4, cudaMemcpyDeviceToHost);
        const float e = std::max(worst(out, out_want), worst(state_after, state_want));
        std::printf("%-42s worst abs %.3g\n", "conv4_silu batch vs a row a call", e);
        if (e > 1e-4f) ++bad;
        cudaFree(d_x); cudaFree(d_w); cudaFree(d_b); cudaFree(d_s); cudaFree(d_o);
    }
    if (bad == 0)
        std::puts("mamba2_step: GPU recurrence matches the host, a chunked prompt matches one call, "
                  "gather_rows copies a projection's parts exactly, and a batch's convolution matches a row a call");
    return bad == 0 ? 0 : 1;
}
