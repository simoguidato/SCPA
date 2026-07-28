#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "kernel.h"
#include "utils.h"
#include "cuda_device_utils.cuh"

// ═══════════════════════════════════════════════════════════════
//  CONFIGURAZIONE THREAD
//
//  FIX rispetto alla versione originale:
//  row_i viene mappato su threadIdx.X (la dimensione che varia
//  più veloce all'interno del warp in CUDA).
//  Nella versione originale era su threadIdx.y → solo 8 indirizzi
//  distinti per warp invece di 32 → accesso non coalescente.
//
//  blockDim(32, 4):
//    threadIdx.x → row_i  (varia veloce, 32 valori distinti per warp)
//    threadIdx.y → col_k  (varia lento,  4  valori distinti per warp)
//  Accesso d_AT[l*M + row_i]: 32 indirizzi contigui → COALESCENTE ✓
// ═══════════════════════════════════════════════════════════════
#define K4_BDIM_ROW 32   // lungo x → row_i (coalescing)
#define K4_BDIM_K    4   // lungo y → col_k

// ═══════════════════════════════════════════════════════════════
//  HELPER ERRORI CUDA
// ═══════════════════════════════════════════════════════════════
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

// ═══════════════════════════════════════════════════════════════
//  MEMORIA DEVICE PERSISTENTE TRA setup/compute/free
//  (stesso pattern degli altri kernel del progetto)
// ═══════════════════════════════════════════════════════════════
static float *d_AT = NULL;   // trasposta A: N×M sul device
static float *d_X  = NULL;   // X: N×k sul device
static float *d_Y  = NULL;   // Y: M×k sul device

// ═══════════════════════════════════════════════════════════════
//  TRASPOSIZIONE SU HOST
//  Converte A (M×N, row-major) in A_T (N×M, row-major)
//  dove A_T[j][i] = A[i][j]  →  A_T[j*M + i] = A[i*N + j]
// ═══════════════════════════════════════════════════════════════
static void host_transpose(int M, int N, const float* src, float* dst) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            dst[j * M + i] = src[i * N + j];
}

// ═══════════════════════════════════════════════════════════════
//  KERNEL 4: Transposed Coalesced
//
//  Calcola Y_loc (M×k) = A_loc (M×N) * X_loc (N×k)
//  usando A_T = A_loc trasposta (N×M, row-major)
//
//  Accessi memoria:
//    d_AT[l*M + row_i]: 32 thread consecutivi (x) leggono
//                        indirizzi contigui → COALESCENTE ✓
//    d_X[l*k  + col_k]: tutti i thread nel semi-warp (stessa y)
//                        leggono lo stesso valore → BROADCAST ✓
//    d_Y[row_i*k+col_k]: scrittura stride-k su row_i variabile
//                         (stesso pattern degli altri kernel)
// ═══════════════════════════════════════════════════════════════
__global__ void matMultivettoreKernel_v4(int M, int N, int k,
                                          const float* __restrict__ d_AT,
                                          const float* __restrict__ d_X,
                                          float* __restrict__ d_Y)
{
    // FIX: row_i su threadIdx.x (varia veloce → coalescing su d_AT)
    //      col_k su threadIdx.y (varia lento  → broadcast su d_X)
    int row_i = blockIdx.x * blockDim.x + threadIdx.x;  // ∈ [0, M)
    int col_k = blockIdx.y * blockDim.y + threadIdx.y;  // ∈ [0, k)

    if (row_i < M && col_k < k) {
        float sum = 0.0f;

        // d_AT è N×M: d_AT[l][row_i] = d_AT[l*M + row_i] = A[row_i][l]
        // d_X  è N×k: d_X[l][col_k]  = d_X[l*k  + col_k]
        // sum = Σ_l A[row_i][l] * X[l][col_k] = (A*X)[row_i][col_k] ✓
        #pragma unroll 8
        for (int l = 0; l < N; ++l) {
            sum += d_AT[l * M + row_i] * d_X[l * k + col_k];
        }

        d_Y[row_i * k + col_k] = sum;
    }
}

// ═══════════════════════════════════════════════════════════════
//  INTERFACCIA STANDARD (kernel.h)
//  Stesso contratto di tutti gli altri kernel del progetto:
//    setup_device_memory  → trasposizione + H2D + malloc (FUORI dal timer)
//    compute_local_gemm   → solo kernel + sync            (DENTRO il timer)
//    free_device_memory   → D2H + cudaFree                (FUORI dal timer)
// ═══════════════════════════════════════════════════════════════

extern "C"
void setup_device_memory(int M, int N, int k,
                          const float *h_A, const float *h_X)
{
    cuda_select_device_or_die();

    // 1. Trasponi A su host (una volta sola, fuori dal timer)
    float *h_AT = NULL;
    if (posix_memalign((void**)&h_AT, 64,
                       (size_t)N * M * sizeof(float)) != 0) {
        fprintf(stderr, "[K4] Errore alloc h_AT\n");
        exit(EXIT_FAILURE);
    }
    host_transpose(M, N, h_A, h_AT);

    // 2. Alloca memoria device
    CHECK_CUDA(cudaMalloc(&d_AT, (size_t)N * M * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_X,  (size_t)N * k * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Y,  (size_t)M * k * sizeof(float)));

    // 3. Copia H2D (A trasposta + X)
    CHECK_CUDA(cudaMemcpy(d_AT, h_AT,
                          (size_t)N * M * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_X, h_X,
                          (size_t)N * k * sizeof(float),
                          cudaMemcpyHostToDevice));

    free(h_AT);
}

extern "C"
void compute_local_gemm(int M, int N, int k,
                         const float *h_A,   // non usato: dati già sul device
                         const float *h_X,   // non usato: dati già sul device
                         float *h_Y)         // non usato: copia in free_device_memory
{
    // Configura griglia:
    //   x → righe di Y (row_i, coalescente su d_AT)
    //   y → colonne di Y (col_k)
    dim3 block(K4_BDIM_ROW, K4_BDIM_K);                       // 32×4 = 128 thread
    dim3 grid((M + K4_BDIM_ROW - 1) / K4_BDIM_ROW,
              (k + K4_BDIM_K   - 1) / K4_BDIM_K);

    matMultivettoreKernel_v4<<<grid, block>>>(M, N, k, d_AT, d_X, d_Y);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

extern "C"
void compute_local_gemm_naive(int M, int N, int k,
                              const float *h_A, const float *h_X, float *h_Y)
{
    // Il kernel CUDA 4 non implementa una variante naive separata: manteniamo
    // l'API completa senza ricorrere a macro di rinomina durante il linking.
    compute_local_gemm(M, N, k, h_A, h_X, h_Y);
}

extern "C"
void free_device_memory(int M, int N, int k, float *h_Y)
{
    // Copia risultato D2H e libera memoria device
    CHECK_CUDA(cudaMemcpy(h_Y, d_Y,
                          (size_t)M * k * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_AT)); d_AT = NULL;
    CHECK_CUDA(cudaFree(d_X));  d_X  = NULL;
    CHECK_CUDA(cudaFree(d_Y));  d_Y  = NULL;
}
