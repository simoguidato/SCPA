#include "kernel.h"
//#include <omp.h>

void setup_device_memory(int M, int N, int k, const double *A, const double *X) {}
void free_device_memory(int M, int N, int k, double *Y) {}
void compute_local_gemm(int M, int N, int k, const double *A, const double *X, double *Y) {

    // Il nuovo pragma corazzato e professionale:
#pragma omp parallel for default(none) shared(M, N, k, A, X, Y) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double a_val = A[i * N + j];
            for (int p = 0; p < k; p++) {
                Y[i * k + p] += a_val * X[j * k + p];
            }
        }
    }
}