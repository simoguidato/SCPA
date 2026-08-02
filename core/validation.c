#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "validation.h"
#include <math.h>
#include "utils.h"

float compute_relative_error(int size, const float *Y_serial, const float *Y_parallel) {
    float norm_diff = 0.0f;
    float norm_serial = 0.0f;

    for (int i = 0; i < size; i++) {
        float diff = Y_serial[i] - Y_parallel[i];
        norm_diff += diff * diff;
        norm_serial += Y_serial[i] * Y_serial[i];
    }

    if (norm_serial == 0) return sqrtf(norm_diff);
    return sqrtf(norm_diff) / sqrtf(norm_serial);
}

void run_validation(AppArgs args, const float *Y_parallel_global, double parallel_avg_time) {

    double gflops = (parallel_avg_time > 0.0) ? (2.0 * args.M * args.N * args.k) / (parallel_avg_time * 1e9) : 0.0;

    // SE DO_VALIDATE È 0, SALTA TUTTO E STAMPA SOLO I GFLOPS PARALLELI
    if (!args.do_validate) {
        printf("DATA_CSV:%.6f,%.6f,%.2e,%.4f,%.4f\n",
               parallel_avg_time, 0.0, 0.0, gflops, 0.0);
        return;
    }

    printf("Avvio verifica seriale globale...\n");
    float *A_global = allocate_matrix(args.M, args.N);
    float *X_global = allocate_matrix(args.N, args.k);
    float *Y_serial = allocate_matrix(args.M, args.k);

    if (A_global && X_global && Y_serial) {
        for (int i = 0; i < args.M; i++) {
            for (int j = 0; j < args.N; j++) {
                unsigned int seed = 42;
                long long global_idx = (long long)i * args.N + j;
                A_global[i * args.N + j] = (float)((global_idx + seed) % 100) / 7.0f;
            }
        }

        for (int i = 0; i < args.N * args.k; i++) {
            X_global[i] = (float)((i + 42 + 17) % 100) / 13.0f;
        }

        double start_serial = MPI_Wtime();
        compute_serial_gemm(args.M, args.N, args.k, A_global, X_global, Y_serial);
        double serial_time = MPI_Wtime() - start_serial;

        float rel_error = compute_relative_error(args.M * args.k, Y_serial, Y_parallel_global);
        double speedup = (parallel_avg_time > 0.0) ? (serial_time / parallel_avg_time) : 0.0;

        printf("--------------------------------------------------\n");
        if (rel_error < 1e-5) {
            printf("[Validazione] Corretto (Errore Relativo: %.2e)\n", rel_error);
        } else {
            printf("[ERRORE] Instabilita numerica! (Errore Relativo: %.2e)\n", rel_error);
        }
        printf("[Benchmark Seriale] Tempo: %.4f sec\n", serial_time);
        printf("[Risultato Finale] SPEEDUP: %.2fx\n", speedup);
        printf("--------------------------------------------------\n");

        printf("DATA_CSV:%.6f,%.6f,%.2e,%.4f,%.4f\n",
               parallel_avg_time, serial_time, rel_error, gflops, speedup);

        free(A_global); free(X_global); free(Y_serial);
    }
}