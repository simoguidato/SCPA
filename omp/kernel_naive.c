#include "kernel.h"
#include <omp.h>

void compute_local_gemm_naive(int M, int N, int k, const double *A, const double *X, double *Y) {
    // Parallelismo base sulle righe
#pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double a_val = A[i * N + j];
            for (int p = 0; p < k; p++) {
                // ACCESSO DIRETTO IN MEMORIA: Molto più lento perché non usa i registri ZMM
                // e causa continue scritture/letture sulla cache L1/L2
                Y[i * k + p] += a_val * X[j * k + p];
            }
        }
    }
}
