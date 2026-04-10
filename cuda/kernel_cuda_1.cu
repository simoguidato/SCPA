#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// =========================================================================
// KERNEL 1: Griglia 2D + Registro Locale + Coalescing su X
//
// Miglioramenti rispetto al Kernel 0 (1D, un thread per riga):
//   - Ogni thread calcola UN SINGOLO elemento Y[row][col], eliminando
//     il loop interno su k dalla versione base.
//   - L'accumulatore 'sum' vive nei registri della GPU: zero scritture
//     in VRAM durante il loop su j, una sola scrittura finale.
//   - Coalescing naturale su X: thread con 'col' contigua leggono
//     X[j*k + col] da indirizzi contigui (transazione unica da 128B).
//   - Broadcast su A: tutti i thread di un Warp (stessa 'row') leggono
//     lo stesso A[row*N + j], attivando il broadcast hardware L1.
// =========================================================================
__global__ void gemm_cuda_kernel_1(int M, int N, int k,
                                   const double *A,
                                   const double *X,
                                   double *Y) {
    // Mappatura 2D: threadIdx.x → colonna di Y (dimensione k)
    //               threadIdx.y → riga di Y    (dimensione M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // indice p in [0, k)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // indice i in [0, M)

    if (row >= M || col >= k) return;

    // Accumulatore locale: il compilatore NVCC lo mappa su un registro
    double sum = 0.0;

    for (int j = 0; j < N; j++) {
        // A[row*N + j]: tutti i thread del Warp leggono lo stesso valore
        //               → broadcast hardware dalla cache L1
        // X[j*k + col]: thread contigui (col=0,1,...,31) leggono indirizzi
        //               contigui → accesso coalescente, un'unica transazione
        sum += A[row * N + j] * X[j * k + col];
    }

    // Scrittura finale: una sola transazione in VRAM per thread
    Y[row * k + col] = sum;
}

// =========================================================================
// INTERFACCIA
// Struttura identica al Kernel 0:
//   setup_device_memory  → alloca e trasferisce A e X sulla GPU (una volta)
//   compute_local_gemm   → lancia il kernel e sincronizza (misurato dal timer)
//   free_device_memory   → copia Y dalla GPU alla CPU e libera la VRAM
// =========================================================================

static double *d_A = NULL;
static double *d_X = NULL;
static double *d_Y = NULL;

void setup_device_memory(int M, int N, int k,
                         const double *A, const double *X) {
    size_t size_A = (size_t)M * N * sizeof(double);
    size_t size_X = (size_t)N * k * sizeof(double);
    size_t size_Y = (size_t)M * k * sizeof(double);

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_X, size_X);
    cudaMalloc((void **)&d_Y, size_Y);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, size_X, cudaMemcpyHostToDevice);
    // Y viene azzerata all'interno di compute_local_gemm
}

void compute_local_gemm(int M, int N, int k,
                        const double *A, const double *X, double *Y) {
    // Azzeramento preventivo di Y sulla GPU
    cudaMemset(d_Y, 0, (size_t)M * k * sizeof(double));

    // Griglia 2D:
    //   - dim x mappa su k  (max 32): un Warp intero copre tutto k in un colpo
    //   - dim y mappa su M:           8 righe per blocco → 32*8 = 256 thread/blocco
    dim3 threadsPerBlock(32, 8);
    dim3 blocksPerGrid(
        (k + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    gemm_cuda_kernel_1<<<blocksPerGrid, threadsPerBlock>>>(M, N, k, d_A, d_X, d_Y);

    // Controllo errori di lancio
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[CUDA Kernel 1] Errore lancio: %s\n", cudaGetErrorString(err));
    }

    // Sincronizzazione: il timer MPI nel main misura fino a qui
    cudaDeviceSynchronize();
}

void free_device_memory(int M, int N, int k, double *Y) {
    // Il cudaMemcpy è FUORI dal timer: la consegna esclude il trasferimento dati
    cudaMemcpy(Y, d_Y, (size_t)M * k * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_X);
    cudaFree(d_Y);

    d_A = NULL;
    d_X = NULL;
    d_Y = NULL;
}
