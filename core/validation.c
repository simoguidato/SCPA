#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "validation.h"
#include "utils.h"

void run_validation(AppArgs args,int rank, const double *Y_parallel_global, double parallel_avg_time) {
    if (args.M > 2000 || args.N > 2000) {
        if (rank == 0) {
            printf("\n[Warning] Matrici troppo grandi (%d x %d). Validazione seriale disattivata per evitare tempi di attesa eccessivi.\n", args.M, args.N);
        }
        return;
    } // Salta il test in silenzio

    printf("[Test] Avvio verifica seriale globale e benchmark base...\n");
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
        double end_serial = MPI_Wtime();

        double serial_time = end_serial - start_serial;
        double serial_gflops = (2.0 * args.M * args.N * args.k) / (serial_time * 1e9);

        if (verify_result(Y_parallel_global, Y_serial, args.M * args.k)) {
            printf("[SUCCESSO] Risultato Parallelo = Risultato Seriale!\n");
        } else {
            printf("[ERRORE] I risultati non coincidono.\n");
        }

        printf("--------------------------------------------------\n");
        printf("[Benchmark Seriale] Tempo: %.4f sec | Prestazioni: %.2f GFLOPS\n", serial_time, serial_gflops);
        if (parallel_avg_time > 0) {
            double speedup = serial_time / parallel_avg_time;
            printf("[Risultato Finale] SPEEDUP: %.2fx\n", speedup);
        }
        printf("--------------------------------------------------\n");

        free(A_global); free(X_global); free(Y_serial);
    }
}