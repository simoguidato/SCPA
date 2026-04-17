#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define TILE_SIZE 32

// =========================================================================
// KERNEL 3: Shared Memory Tiling + Texture/Read-Only Cache (restrict)
// =========================================================================
__global__ void gemm_cuda_tiled(int M, int N, int k, const double* __restrict__ A, const double* __restrict__ X, double *Y) {

    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Xs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    double sum = 0.0;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int m = 0; m < numTiles; m++) {
        // Lettura con caching ottimizzato dalla VRAM alla Shared Memory
        if (row < M && (m * TILE_SIZE + tx) < N) {
            As[ty][tx] = __ldg(&A[row * N + (m * TILE_SIZE + tx)]);
        } else {
            As[ty][tx] = 0.0;
        }

        if ((m * TILE_SIZE + ty) < N && col < k) {
            Xs[ty][tx] = __ldg(&X[(m * TILE_SIZE + ty) * k + col]);
        } else {
            Xs[ty][tx] = 0.0;
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
static double *d_A = NULL;
static double *d_X = NULL;
static double *d_Y = NULL;

void setup_device_memory(int M, int N, int k, const double *A, const double *X) {
    size_t size_A = (size_t)M * N * sizeof(double);
    size_t size_X = (size_t)N * k * sizeof(double);
    size_t size_Y = (size_t)M * k * sizeof(double);

    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_X, size_X));
    CHECK_CUDA(cudaMalloc((void **)&d_Y, size_Y));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice));
}

// ... nel file kernel_cuda_2.cu ...

void compute_local_gemm(int M, int N, int k, const double *A, const double *X, double *Y) {
    // Rimuoviamo la cudaMemcpy da qui!
    CHECK_CUDA(cudaMemset(d_Y, 0, (size_t)M * k * sizeof(double)));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((k + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_cuda_tiled<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);
    CHECK_CUDA(cudaGetLastError());
    // Sincronizziamo per essere sicuri che il calcolo sia finito prima di fermare il timer nel main
    CHECK_CUDA(cudaDeviceSynchronize());
}

void free_device_memory(int M, int N, int k, double *Y) {
    // La copia dei dati avviene QUI, dopo che il timer nel main si è fermato
    size_t size_Y = (size_t)M * k * sizeof(double);
    CHECK_CUDA(cudaMemcpy(Y, d_Y, size_Y, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}