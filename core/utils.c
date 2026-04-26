#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "utils.h"
#include <math.h>

float epsilon = 1e-5f;
// Allocazione allineata a 64 byte per massimizzare MFLOPS [punto 24 e 40]
float *allocate_matrix(int rows, int cols) {
    float *ptr = NULL;
    size_t size = (size_t)rows * cols * sizeof(float);
    if (size == 0) return NULL;

    if (posix_memalign((void**)&ptr, 64, size) != 0) {
        fprintf(stderr, "Errore: Memoria insufficiente\n");
        return NULL;
    }
    return ptr;
}

// Generazione locale ripetibile senza lo spreco di srand nel loop [punto 17]
void generate_data_locally(float *local_A, GridInfo info, unsigned int seed) {
    for (int i = 0; i < info.local_M; i++) {
        for (int j = 0; j < info.local_N; j++) {
            // Formula deterministica basata su posizione globale
            long long global_idx = (long long)(info.offset_M + i) * info.global_N + (info.offset_N + j);
            local_A[i * info.local_N + j] = (float)((global_idx + seed) % 100) / 7.0f;
        }
    }
}

// Verifica del risultato (essenziale per il collaudo [punto 12])
int verify_result(const float *Y_parallel, const float *Y_serial, int size) {
    for (int i = 0; i < size; i++) {
        if (fabsf(Y_parallel[i] - Y_serial[i]) > epsilon) {
            return 0; // Errore nel calcolo
        }
    }
    return 1; // Successo
}
// Calcola Y = A * X in modo seriale per verificare la correttezza
void compute_serial_gemm(int M, int N, int k, const float *A, const float *X, float *Y) {
    // Inizializza Y a zero
    for (int i = 0; i < M * k; i++) {
        Y[i] = 0.0f;
    }

    // Calcolo Row-Major
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float a_val = A[i * N + j];
            for (int p = 0; p < k; p++) {
                Y[i * k + p] += a_val * X[j * k + p];
            }
        }
    }
}

// Generazione deterministica di X basata sull'offset globale delle colonne di A
void generate_X_locally(float *local_X, int local_N, int k, int offset_N, unsigned int seed) {
    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < k; j++) {
            long long global_idx = (long long)(offset_N + i) * k + j;
            // Generiamo numeri decimali deterministici
            local_X[i * k + j] = (float)((global_idx + seed + 17) % 100) / 13.0f;
            //local_X[i * k + j] = 1.0;    PER FARE TEST
        }
    }
}

