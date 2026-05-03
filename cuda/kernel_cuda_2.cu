#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 32

// =========================================================================
// KERNEL 3: Shared Memory Tiling + Texture/Read-Only Cache (restrict)
// =========================================================================
__global__ void gemm_cuda_tiled(int M, int N, int k, const float* __restrict__ A, const float* __restrict__ X, float *Y) {

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Xs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int m = 0; m < numTiles; m++) {
        // Lettura con caching ottimizzato dalla VRAM alla Shared Memory
        if (row < M && (m * TILE_SIZE + tx) < N) {
            As[ty][tx] = __ldg(&A[row * N + (m * TILE_SIZE + tx)]);
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((m * TILE_SIZE + ty) < N && col < k) {
            Xs[ty][tx] = __ldg(&X[(m * TILE_SIZE + ty) * k + col]);
        } else {
            Xs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Xs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < k) {
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
    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_X, size_X));
    CHECK_CUDA(cudaMalloc((void **)&d_Y, size_Y));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice));
}

void compute_local_gemm(int M, int N, int k, const float *A, const float *X, float *Y) {
    CHECK_CUDA(cudaMemset(d_Y, 0, (size_t)M * k * sizeof(float)));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((k + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_cuda_tiled<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);
    CHECK_CUDA(cudaGetLastError());
    // Sincronizziamo per essere sicuri che il calcolo sia finito prima di fermare il timer nel main
    CHECK_CUDA(cudaDeviceSynchronize());
}

void free_device_memory(int M, int N, int k, float *Y) {
    size_t size_Y = (size_t)M * k * sizeof(float);
    CHECK_CUDA(cudaMemcpy(Y, d_Y, size_Y, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}