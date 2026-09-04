#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include "utils.h"
#include "cuda_device_utils.cuh"
// ═══════════════════════════════════════════════════════════════
//  CONFIGURAZIONE THREAD
//  Utilizziamo un Tile quadrato 32x32 per ottimizzare sia la
//  lettura coalescente (su AT) che la scrittura coalescente (su Y).
// ═══════════════════════════════════════════════════════════════
#define TILE_DIM 32
#undef CHECK_CUDA
#define CHECK_CUDA(val) check_cuda_err((val), #val, __FILE__, __LINE__)
static void check_cuda_err(cudaError_t err, const char* func,
                            const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d — %s: %s\n",
                file, line, func, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Puntatori globali per mantenere i dati tra le fasi
static float *d_A  = NULL; // A originale (temporanea su GPU)
static float *d_AT = NULL; // A trasposta su GPU
static float *d_X  = NULL; // Multivettore X
static float *d_Y  = NULL; // Risultato Y

__global__ void transpose_gpu_kernel(int M, int N, const float* __restrict__ idata, float* __restrict__ odata) {
    // +1 per azzerare i bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Lettura coalescente dalla matrice originale
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = idata[y * N + x];
    }

    __syncthreads();

    // Ricalcolo coordinate per la scrittura trasposta
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Scrittura coalescente sulla matrice destinazione
    if (x < M && y < N) {
        odata[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void matMultivettoreKernel_v4(int M, int N, int k,
                                          const float* __restrict__ d_AT,
                                          const float* __restrict__ d_X,
                                          float* __restrict__ d_Y)
{
    __shared__ float smem_Y[TILE_DIM][TILE_DIM + 1];

    // Mappatura 1: threadIdx.x associato a row_i per avere lettura
    // perfettamente coalescente da d_AT
    int row_i = blockIdx.x * TILE_DIM + threadIdx.x;
    int col_k = blockIdx.y * TILE_DIM + threadIdx.y;

    float sum = 0.0f;
    if (row_i < M && col_k < k) {
        #pragma unroll 8
        for (int l = 0; l < N; ++l) {
            sum += d_AT[l * M + row_i] * d_X[l * k + col_k];
        }
    }

    // Salvataggio temporaneo in Shared Memory.
    // Thread del warp (stesso threadIdx.y) scrivono su threadIdx.x consecutivi.
    if (row_i < M && col_k < k) {
        smem_Y[threadIdx.y][threadIdx.x] = sum;
    }

    __syncthreads();

    // Mappatura 2: Invertiamo la logica. Ora threadIdx.x (che garantisce
    // la coalescenza fisica del Warp) viene associato a col_k!
    int out_col_k = blockIdx.y * TILE_DIM + threadIdx.x;
    int out_row_i = blockIdx.x * TILE_DIM + threadIdx.y;

    if (out_row_i < M && out_col_k < k) {
        // Lettura dalla Shared Memory e scrittura su RAM globale
        // d_Y layout: righe x colonne.
        d_Y[out_row_i * k + out_col_k] = smem_Y[threadIdx.x][threadIdx.y];
    }
}

extern "C"
void setup_device_memory(int M, int N, int k,
                          const float *h_A, const float *h_X)
{
    cuda_select_device_or_die();

    // Alloca memorie sul device
    CHECK_CUDA(cudaMalloc(&d_A,  (size_t)N * M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_AT, (size_t)N * M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_X,  (size_t)N * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y,  (size_t)M * k * sizeof(float)));

    //  Trasferimento H2D
    CHECK_CUDA(cudaMemcpy(d_A, h_A, (size_t)N * M * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, (size_t)N * k * sizeof(float), cudaMemcpyHostToDevice));

    // Trasposizione in VRAM
    dim3 block_trans(TILE_DIM, TILE_DIM);
    dim3 grid_trans((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    transpose_gpu_kernel<<<grid_trans, block_trans>>>(M, N, d_A, d_AT);
    CHECK_CUDA(cudaDeviceSynchronize());

    // La matrice A non trasposta non serve più, liberiamo VRAM
    CHECK_CUDA(cudaFree(d_A));
    d_A = NULL;
}

extern "C"
void compute_local_gemm(int M, int N, int k,
                         const float *h_A,
                         const float *h_X,
                         float *h_Y)
{
    // Griglia bloccata a 32x32
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (k + TILE_DIM - 1) / TILE_DIM);

    matMultivettoreKernel_v4<<<grid, block>>>(M, N, k, d_AT, d_X, d_Y);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

extern "C"
void compute_local_gemm_naive(int M, int N, int k,
                              const float *h_A, const float *h_X, float *h_Y)
{
    compute_local_gemm(M, N, k, h_A, h_X, h_Y);
}

extern "C"
void free_device_memory(int M, int N, int k, float *h_Y)
{
    CHECK_CUDA(cudaMemcpy(h_Y, d_Y, (size_t)M * k * sizeof(float), cudaMemcpyDeviceToHost));

    // Rilascio di sicurezza per tutte le strutture
    if(d_A)  { CHECK_CUDA(cudaFree(d_A));  d_A  = NULL; }
    if(d_AT) { CHECK_CUDA(cudaFree(d_AT)); d_AT = NULL; }
    if(d_X)  { CHECK_CUDA(cudaFree(d_X));  d_X  = NULL; }
    if(d_Y)  { CHECK_CUDA(cudaFree(d_Y));  d_Y  = NULL; }
}