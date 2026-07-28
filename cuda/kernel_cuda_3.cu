#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_device_utils.cuh"

// =========================================================================
// KERNEL 4: Warp-Row / Register Tiling (Zero Shared Memory Sync)
// =========================================================================
__global__ void gemm_warp_row_register(int M, int N, int k, const float* __restrict__ A, const float* __restrict__ X, float* Y) {
    // Un warp (32 thread) si occupa di una intera riga di A.
    // blockDim.y indica quanti warp ci sono in un blocco.
    int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    int lane_id = threadIdx.x; // Indice del thread all'interno del warp (da 0 a 31)

    // Se questo warp è assegnato a una riga valida
    if (warp_id < M) {
        // Se k > 32, il thread si occuperà di più colonne saltando di 32 in 32
        for (int p = lane_id; p < k; p += 32) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                // Tutti i 32 thread del warp leggono lo STESSO elemento di A.
                float a_val = A[warp_id * N + j];
                // Ogni thread moltiplica per la sua specifica colonna di X
                sum += a_val * X[j * k + p];
            }
            Y[warp_id * k + p] = sum;
        }
    }
}

__global__ void gemm_naive_2d(int M, int N, int k, const float *A,
                              const float *X, float *Y) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < k) {
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            sum += A[row * N + j] * X[j * k + col];
        }
        Y[row * k + col] = sum;
    }
}

// =========================================================================
// INTERFACCIA CPU-GPU
// =========================================================================
static float *d_A = NULL;
static float *d_X = NULL;
static float *d_Y = NULL;

void setup_device_memory(int M, int N, int k, const float *A, const float *X) {
    size_t size_A = (size_t)M * N * sizeof(float);
    size_t size_X = (size_t)N * k * sizeof(float);
    size_t size_Y = (size_t)M * k * sizeof(float);
    cuda_select_device_or_die();
    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_X, size_X));
    CHECK_CUDA(cudaMalloc((void **)&d_Y, size_Y));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice));
}

void compute_local_gemm(int M, int N, int k, const float *A, const float *X, float *Y) {
    // CONFIGURAZIONE DELLA GRIGLIA "A WARP"
    // blocchi 2D: X=32 (la larghezza fissa di un warp), Y=8 (8 warp per blocco)
    // Totale thread per blocco = 256.
    dim3 threadsPerBlock(32, 8);
    // Ogni blocco copre 8 righe (perché ha 8 warp)
    dim3 blocksPerGrid((M + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    gemm_warp_row_register<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void compute_local_gemm_naive(int M, int N, int k, const float *A, const float *X, float *Y) {
    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid((k + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    gemm_naive_2d<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void free_device_memory(int M, int N, int k, float *Y) {
    CHECK_CUDA(cudaMemcpy(Y, d_Y, (size_t)M * k * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}
