// relu2_rows(), used by Nemotron-H's single-projection experts, against the host expression on a real GPU.
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr int n = 7, rows = 2;
    const std::vector<float> x{-4.f, -0.f, 0.25f, 1.f, 3.f, -2.f, 0.5f,
                               8.f, -7.f, 2.f, -0.5f, 0.f, 1.5f, -1.f};
    float *dx = nullptr, *dy = nullptr;
    if (cudaMalloc(&dx, x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dy, x.size() * sizeof(float)) != cudaSuccess) return 2;
    if (cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
    strata::poly::relu2_rows(dx, dy, n, rows, nullptr);
    std::vector<float> y(x.size());
    if (cudaMemcpy(y.data(), dy, y.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 2;
    cudaFree(dy);
    cudaFree(dx);
    for (size_t i = 0; i < x.size(); ++i) {
        const float r = x[i] > 0.f ? x[i] : 0.f;
        if (y[i] != r * r) {
            std::fprintf(stderr, "relu2 mismatch at %zu: got %.9g, expected %.9g\n", i, y[i], r * r);
            return 1;
        }
    }
    std::puts("relu2_rows: 14/14 values match");
    return 0;
}
