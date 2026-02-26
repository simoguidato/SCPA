#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include "utils.h"

// Allocazione allineata a 64 byte per massimizzare MFLOPS [punto 24 e 40]
double* allocate_matrix(int rows, int cols) {
    double *ptr = NULL;
    size_t size = (size_t)rows * cols * sizeof(double);
    if (size == 0) return NULL;

    if (posix_memalign((void**)&ptr, 64, size) != 0) {
        fprintf(stderr, "Errore: Memoria insufficiente\n");
        return NULL;
    }
    return ptr;
}

// Generazione locale ripetibile senza lo spreco di srand nel loop [punto 17]
void generate_data_locally(double *local_A, GridInfo info, unsigned int seed) {
    for (int i = 0; i < info.local_M; i++) {
        for (int j = 0; j < info.local_N; j++) {
            // Formula deterministica basata su posizione globale
            long long global_idx = (long long)(info.offset_M + i) * info.global_N + (info.offset_N + j);
            local_A[i * info.local_N + j] = (double)((global_idx + seed) % 100) / 7.0;
        }
    }
}

// Verifica del risultato (essenziale per il collaudo [punto 12])
int verify_result(double *Y_parallel, double *Y_serial, int size) {
    double epsilon = 1e-9;
    for (int i = 0; i < size; i++) {
        if (abs(Y_parallel[i] - Y_serial[i]) > epsilon) {
            return 0; // Errore nel calcolo
        }
    }
    return 1; // Successo
}