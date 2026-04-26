#include "kernel.h"
#include <assert.h>
#include <string.h>

// Dimensione perfetta per il Working Set di 24KB (entra tutto nei 32KB della L1)
//#define BLOCK_SIZE 32
#define MAX_K 32
void setup_device_memory(int M, int N, int k, const float *A, const float *X) {}
void free_device_memory(int M, int N, int k, float *Y) {}

void compute_local_gemm(int M, int N, int k, const float *A, const float *X, float *Y) {

    // Programmazione difensiva: crasha in modo pulito e immediato se
    // l'utente inserisce da terminale un k non supportato dalle specifiche.
    assert(k <= MAX_K && "Errore: k supera la dimensione massima prevista (MAX_K=32)");

    // Parallelismo puro sulle righe M.
    // Nessuna barriera, NUMA affinity garantita dal pattern First-Touch.
#pragma omp parallel for schedule(static) default(none) shared(M, N, k, A, X, Y)
    for (int i = 0; i < M; i++) {
        // Array a dimensione STATICA.
        // Il compilatore ora sa esattamente quanto è grande e lo mappa sui registri ZMM.
        float y_local[MAX_K] = {0};

        for (int j = 0; j < N; j++) {
            float a_val = A[i * N + j];
#pragma omp simd safelen(MAX_K)
            for ( int p = 0; p < k; p++) {
                y_local[p] += a_val * X[j * k + p];
            }
        }

#pragma omp simd safelen(MAX_K)
        for (int s = 0; s < k; s++) {
            Y[i * k + s] = y_local[s];
        }
    }
}