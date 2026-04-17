#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// =========================================================================
// 1. IL KERNEL GPU
// =========================================================================
__global__ void gemm_cuda_kernel(int M, int N, int k, const double *A, const double *X, double *Y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        for (int j = 0; j < N; j++) {
            double a_val = A[i * N + j];
            for (int p = 0; p < k; p++) {
                Y[i * k + p] += a_val * X[j * k + p];
            }
        }
    }
}

// =========================================================================
// 2. L'INTERFACCIA CPU-GPU
// =========================================================================
static double *d_A = NULL;
static double *d_X = NULL;
static double *d_Y = NULL;

void setup_device_memory(int M, int N, int k, const double *A, const double *X) {
    size_t size_A = (size_t)M * N * sizeof(double);
    size_t size_X = (size_t)N * k * sizeof(double);
    size_t size_Y = (size_t)M * k * sizeof(double);

    // Allocazione e Copia Protette
    CHECK_CUDA(cudaMalloc((void**)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void**)&d_X, size_X));
    CHECK_CUDA(cudaMalloc((void**)&d_Y, size_Y));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice));
}

void compute_local_gemm(int M, int N, int k, const double *A, const double *X, double *Y) {
    CHECK_CUDA(cudaMemset(d_Y, 0, (size_t)M * k * sizeof(double)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Lancio asincrono
    gemm_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);

    // Controllo Errori di Lancio ed Esecuzione
    CHECK_CUDA(cudaGetLastError());
    // Sincronizziamo la GPU per fermare correttamente il timer nel main!
    CHECK_CUDA(cudaDeviceSynchronize());

    // RIMOSSA LA CUDAMEMCPY DA QUI
}

void free_device_memory(int M, int N, int k, double *Y) {
    // IL TRASFERIMENTO (D2H) AVVIENE QUI, FUORI DAL TIMER
    CHECK_CUDA(cudaMemcpy(Y, d_Y, (size_t)M * k * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}
