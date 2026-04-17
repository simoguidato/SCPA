#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// =========================================================================
// KERNEL 4: Warp-Row / Register Tiling (Zero Shared Memory Sync)
// =========================================================================
__global__ void gemm_warp_row_register(int M, int N, int k, const double* __restrict__ A, const double* __restrict__ X, double* Y) {
    // Un warp (32 thread) si occupa di una intera riga di A.
    // blockDim.y indica quanti warp ci sono in un blocco.
    int warp_id = (blockIdx.x * blockDim.y) + threadIdx.y;
    int lane_id = threadIdx.x; // Indice del thread all'interno del warp (da 0 a 31)

    // Se questo warp è assegnato a una riga valida
    if (warp_id < M) {
        // Se k > 32, il thread si occuperà di più colonne saltando di 32 in 32
        for (int p = lane_id; p < k; p += 32) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                // Tutti i 32 thread del warp leggono lo STESSO elemento di A.
                // L'hardware GPU lo trasforma in un "Broadcast" ultraveloce.
                double a_val = A[warp_id * N + j];

                // Ogni thread moltiplica per la sua specifica colonna di X
                sum += a_val * X[j * k + p];
            }
            Y[warp_id * k + p] = sum;
        }
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

void compute_local_gemm(int M, int N, int k, const double *A, const double *X, double *Y) {
    CHECK_CUDA(cudaMemset(d_Y, 0, (size_t)M * k * sizeof(double)));

    // CONFIGURAZIONE DELLA GRIGLIA "A WARP"
    // Vogliamo blocchi 2D: X=32 (la larghezza fissa di un warp), Y=8 (8 warp per blocco)
    // Totale thread per blocco = 256.
    dim3 threadsPerBlock(32, 8);

    // Quanti blocchi servono per coprire le M righe?
    // Ogni blocco copre 8 righe (perché ha 8 warp)
    dim3 blocksPerGrid((M + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

    // Lancio del kernel
    gemm_warp_row_register<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);

    // CONTROLLI DI SICUREZZA (Ora ci sono!)
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ALIAS PER IL MAIN.C
void compute_local_gemm_naive(int M, int N, int k, const double *A, const double *X, double *Y) {
    compute_local_gemm(M, N, k, A, X, Y);
}

void free_device_memory(int M, int N, int k, double *Y) {
    // Trasferimento D2H isolato dal timer!
    CHECK_CUDA(cudaMemcpy(Y, d_Y, (size_t)M * k * sizeof(double), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
}