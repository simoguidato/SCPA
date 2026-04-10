#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "validation.h"
#include <math.h>
#include "utils.h"

double compute_relative_error(int size, const double *Y_serial, const double *Y_parallel) {
    double norm_diff = 0.0;
    double norm_serial = 0.0;

    for (int i = 0; i < size; i++) {
        double diff = Y_serial[i] - Y_parallel[i];
        norm_diff += diff * diff;
        norm_serial += Y_serial[i] * Y_serial[i];
    }

    if (norm_serial == 0) return sqrt(norm_diff);
    return sqrt(norm_diff) / sqrt(norm_serial);
}

void run_validation(AppArgs args, const double *Y_parallel_global, double parallel_avg_time) {

    double gflops = (parallel_avg_time > 0) ? (2.0 * args.M * args.N * args.k) / (parallel_avg_time * 1e9) : 0.0;

    // SE DO_VALIDATE È 0, SALTA TUTTO E STAMPA SOLO I GFLOPS PARALLELI
    if (!args.do_validate) {
        // Stampiamo 0.0 per i tempi seriali, Python gestirà il ricalcolo
        printf("DATA_CSV:%.6f,%.6f,%.2e,%.4f,%.4f\n",
               parallel_avg_time, 0.0, 0.0, gflops, 0.0);
        return;
    }

    printf("[Test] Avvio verifica seriale globale...\n");
    double *A_global = allocate_matrix(args.M, args.N);
    double *X_global = allocate_matrix(args.N, args.k);
    double *Y_serial = allocate_matrix(args.M, args.k);

    if (A_global && X_global && Y_serial) {
        for (int i = 0; i < args.M; i++) {
            for (int j = 0; j < args.N; j++) {
                unsigned int seed = 42;
                long long global_idx = (long long)i * args.N + j;
                A_global[i * args.N + j] = (double)((global_idx + seed) % 100) / 7.0;
            }
        }

        for (int i = 0; i < args.N * args.k; i++) {
            X_global[i] = (double)((i + 42 + 17) % 100) / 13.0;
        }

        double start_serial = MPI_Wtime();
        compute_serial_gemm(args.M, args.N, args.k, A_global, X_global, Y_serial);
        double serial_time = MPI_Wtime() - start_serial;

        double rel_error = compute_relative_error(args.M * args.k, Y_serial, Y_parallel_global);
        double speedup = (parallel_avg_time > 0) ? (serial_time / parallel_avg_time) : 0.0;

        printf("--------------------------------------------------\n");
        if (rel_error < 1e-10) {
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