#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// =========================================================================
// 1. IL KERNEL GPU
// =========================================================================
__global__ void gemm_cuda_kernel(int M, int N, int k, const double *A, const double *X, double *Y) {
    // Calcoliamo l'indice globale del thread (che corrisponderà alla riga 'i')
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Controllo di sicurezza: se abbiamo lanciato più thread delle righe effettive,
    // i thread in eccesso non fanno nulla (escono).
    if (i < M) {
        // Ogni thread esegue esattamente lo stesso calcolo della versione OpenMP,
        // ma SOLO per la sua specifica riga 'i'
        for (int j = 0; j < N; j++) {
            double a_val = A[i * N + j];
            for (int p = 0; p < k; p++) {
                Y[i * k + p] += a_val * X[j * k + p];
            }
        }
    }
}

// =========================================================================
// 2. L'INTERFACCIA (Chiamata dal main.c, eseguita dalla CPU)
// =========================================================================
static double *d_A = NULL;
static double *d_X = NULL;
static double *d_Y = NULL;

void setup_device_memory(int M, int N, int k, const double *A, const double *X) {
    size_t size_A = (size_t)M * N * sizeof(double);
    size_t size_X = (size_t)N * k * sizeof(double);
    size_t size_Y = (size_t)M * k * sizeof(double);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_X, size_X);
    cudaMalloc((void**)&d_Y, size_Y);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice);
    // Non copiamo Y perché lo azzereremo direttamente nella GPU
}

void compute_local_gemm(int M, int N, int k, const double *A, const double *X, double *Y) {
    // 1. Azzeriamo Y direttamente sulla memoria ultra-veloce della GPU
    cudaMemset(d_Y, 0, M * k * sizeof(double));

    // 2. Lanciamo il calcolo
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;
    gemm_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(...);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("[CUDA Kernel 0] Errore: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}

void free_device_memory(int M, int N, int k, double *Y) {
    // Riportiamo il risultato calcolato dalla VRAM alla RAM della CPU
    cudaMemcpy(Y, d_Y, M * k * sizeof(double), cudaMemcpyDeviceToHost);

    // Puliamo la memoria della scheda video
    cudaFree(d_A); d_A = NULL;
    cudaFree(d_X); d_X = NULL;
    cudaFree(d_Y); d_Y = NULL;
}
