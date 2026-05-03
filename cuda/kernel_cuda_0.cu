#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

// =========================================================================
// 1. IL KERNEL GPU
// =========================================================================
__global__ void gemm_cuda_kernel(int M, int N, int k, const float *A, const float *X, float *Y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M) {
        for (int j = 0; j < N; j++) {
            float a_val = A[i * N + j];
            for (int p = 0; p < k; p++) {
                Y[i * k + p] += a_val * X[j * k + p];
            }
        }
    }
}

// =========================================================================
// 2. L'INTERFACCIA CPU-GPU
// =========================================================================
static float *d_A = NULL;
static float *d_X = NULL;
static float *d_Y = NULL;

void setup_device_memory(int M, int N, int k, const float *A, const float *X) {
    size_t size_A = (size_t)M * N * sizeof(float);
    size_t size_X = (size_t)N * k * sizeof(float);
    size_t size_Y = (size_t)M * k * sizeof(float);
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    int rank = 0;
    if (getenv("OMPI_COMM_WORLD_RANK")) {
        rank = atoi(getenv("OMPI_COMM_WORLD_RANK"));
    } else if (getenv("PMI_RANK")) {
        rank = atoi(getenv("PMI_RANK"));
    }

    // Associa il processo alla GPU corretta (es. Rank 0 -> GPU 0, Rank 1 -> GPU 1)
    CHECK_CUDA(cudaSetDevice(rank % num_devices));

    CHECK_CUDA(cudaMalloc((void**)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void**)&d_X, size_X));
    CHECK_CUDA(cudaMalloc((void**)&d_Y, size_Y));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice));
}

void compute_local_gemm(int M, int N, int k, const float *A, const float *X, float *Y) {
    CHECK_CUDA(cudaMemset(d_Y, 0, (size_t)M * k * sizeof(float)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    gemm_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);

    // Controllo Errori di Lancio ed Esecuzione
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

void free_device_memory(int M, int N, int k, float *Y) {
    CHECK_CUDA(cudaMemcpy(Y, d_Y, (size_t)M * k * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}
