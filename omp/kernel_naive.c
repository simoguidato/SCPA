#include "kernel.h"

void compute_local_gemm_naive(int M, int N, int k, const float *A, const float *X, float *Y) {
    // Parallelismo base sulle righe
#pragma omp parallel for schedule(static) default(none) shared(M, N, k, A, X, Y)
    for (int i = 0; i < M; i++) {
        for (int p = 0; p < k; p++) {
            Y[i * k + p] = 0.0f;
        }

        for (int j = 0; j < N; j++) {
            float a_val = A[i * N + j];
            for (int p = 0; p < k; p++) {
                Y[i * k + p] += a_val * X[j * k + p];
            }
        }
    }
}