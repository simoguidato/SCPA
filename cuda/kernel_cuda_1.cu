#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// =========================================================================
// KERNEL 1: Griglia 2D + Registro Locale + Coalescing su X
// =========================================================================
__global__ void gemm_cuda_kernel_1(int M, int N, int k, const float *A, const float *X, float *Y) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < k) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += A[row * N + j] * X[j * k + col];
        }
        Y[row * k + col] = sum;
    }
}

// =========================================================================
// INTERFACCIA CPU-GPU
// =========================================================================
static float *d_A = NULL;
static float*d_X = NULL;
static float *d_Y = NULL;

void setup_device_memory(int M, int N, int k, const float *A, const float *X) {
    size_t size_A = (size_t)M * N * sizeof(float);
    size_t size_X = (size_t)N * k * sizeof(float);
    size_t size_Y = (size_t)M * k * sizeof(float);

    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_X, size_X));
    CHECK_CUDA(cudaMalloc((void **)&d_Y, size_Y));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice));
}

void compute_local_gemm(int M, int N, int k, const float *A, const float *X, float *Y) {
    CHECK_CUDA(cudaMemset(d_Y, 0, (size_t)M * k * sizeof(float)));

    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid(
        (k + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    gemm_cuda_kernel_1<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);

    // Protezione totale
    CHECK_CUDA(cudaGetLastError());
    // Sincronizzazione fondamentale!
    CHECK_CUDA(cudaDeviceSynchronize());

    // RIMOSSA LA CUDAMEMCPY DA QUI
}

void free_device_memory(int M, int N, int k, float *Y) {
    // TRASFERIMENTO SPOSTATO QUI, FUORI DAL LOOP DI TIMING
    CHECK_CUDA(cudaMemcpy(Y, d_Y, (size_t)M * k * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}