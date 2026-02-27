#include "kernel.h"
//#include <omp.h>

void compute_local_gemm(int local_M, int local_N, int k,
                        const double *A_local, const double *X_local,
                        double *Y_local) {

    // Parallelizziamo sulle righe della matrice locale A
#pragma omp parallel for schedule(static)
    for (int i = 0; i < local_M; i++) {
        for (int j = 0; j < local_N; j++) {
            // Leggiamo l'elemento di A una sola volta (ottimizzazione Cache)
            double a_val = A_local[i * local_N + j];

            // Poiché k è piccolo (3, 6, 8...), questo ciclo interno è velocissimo
            // e il compilatore può applicare l'unrolling automatico
            for (int p = 0; p < k; p++) {
                Y_local[i * k + p] += a_val * X_local[j * k + p];
            }
        }
    }
}