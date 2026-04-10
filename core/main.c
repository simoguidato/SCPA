#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "args.h"
#include "mpi_grid.h"
#include "utils.h"
#include "kernel.h"
#include "validation.h"
#include <assert.h>

int main(int argc, char *argv[]) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 1. Setup Argomenti
    AppArgs args = parse_arguments(argc, argv, rank, num_procs);

    // 2. Setup Topologia MPI
    int dims[2] = {0, 0};
    MPI_Dims_create(num_procs, 2, dims);
    int periods[2] = {0, 0};
    MPI_Comm cart_comm, row_comm, col_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    if (args.M < dims[0] || args.N < dims[1]) {
        if (rank == 0) fprintf(stderr, "[Errore] Matrice troppo piccola per la griglia.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    GridInfo info;
    compute_grid_partition(args.M, args.N, dims[0], dims[1], coords[0], coords[1], &info);
    create_sub_communicators(cart_comm, &row_comm, &col_comm);

    // 3. Allocazione e Setup Dati Locali
    double *local_A = allocate_matrix(info.local_M, info.local_N);
    double *local_X = allocate_matrix(info.local_N, args.k);
    double *local_Y = allocate_matrix(info.local_M, args.k);
    if (local_A == NULL || local_X == NULL || local_Y == NULL) {
        fprintf(stderr, "[Errore] Allocazione memoria fallita.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }

    // Tutti generano la loro fetta di A
    generate_data_locally(local_A, info, 42);

    // RISPETTO DELLE SPECIFICHE DEL PROGETTO:
    // "si può assumere che X occupi una sola riga di questa griglia"
    if (coords[0] == 0) {
        // Solo i processi nella PRIMA RIGA della griglia generano X
        generate_X_locally(local_X, info.local_N, args.k, info.offset_N, 42);
    }

    // I processi della prima riga (root = 0 del col_comm) trasmettono X
    // a tutti i processi sottostanti nella loro stessa colonna.
    MPI_Bcast(local_X, info.local_N * args.k, MPI_DOUBLE, 0, col_comm);
    if (rank == 0) printf("[MPI] Multivettore X distribuito.\n");
    // 4. Benchmark e Calcolo
    int num_iter = 10;

    // --> SPOSTIAMO I DATI SULLA GPU (Se usiamo OpenMP, non fa nulla)
    setup_device_memory(info.local_M, info.local_N, args.k, local_A, local_X);

    if (args.do_warmup) {
        if (args.kernel_type == 0) {
            compute_local_gemm(info.local_M, info.local_N, args.k, local_A, local_X, local_Y);
        } else {
            compute_local_gemm_naive(info.local_M, info.local_N, args.k, local_A, local_X, local_Y);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    for (int it = 0; it < num_iter; it++) {
        if (args.kernel_type == 0) {
            compute_local_gemm(info.local_M, info.local_N, args.k, local_A, local_X, local_Y);
        } else {
            compute_local_gemm_naive(info.local_M, info.local_N, args.k, local_A, local_X, local_Y);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    // --> RIPORTIAMO I DATI SULLA CPU
    free_device_memory(info.local_M, info.local_N, args.k, local_Y);

    // Calcolo tempi e GFLOPS...
    double local_time = end_time - start_time;
    double max_global_time = 0.0;
    MPI_Reduce(&local_time, &max_global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double parallel_avg_time = 0.0;
    if (rank == 0) {
        parallel_avg_time = max_global_time / num_iter;
        double gflops = (2.0 * args.M * args.N * args.k) / (parallel_avg_time * 1e9);
        printf("[Benchmark Parallelo] Tempo medio: %.4f sec | Prestazioni: %.2f GFLOPS\n", parallel_avg_time, gflops);
    }

    // 5. Raccolta Dati (Gather)
    double *row_Y = NULL;
    if (coords[1] == 0) {
        row_Y = allocate_matrix(info.local_M, args.k);
        if (row_Y == NULL) {
            fprintf(stderr, "[Errore] Allocazione fallita per row_Y.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            exit(EXIT_FAILURE);
        }
    }

    // Somma dei risultati parziali lungo la riga
    MPI_Reduce(local_Y, row_Y, info.local_M * args.k, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    double *Y_parallel_global = NULL;
    int *recvcounts = NULL, *displs = NULL;

    if (coords[1] == 0) {
        int my_Y_size = info.local_M * args.k;

        // 1. Il Master alloca TUTTO subito (così l'IDE non si confonde)
        if (rank == 0) {
            recvcounts = malloc(dims[0] * sizeof(int));
            displs = malloc(dims[0] * sizeof(int));
            Y_parallel_global = allocate_matrix(args.M, args.k);

            if (recvcounts == NULL || displs == NULL || Y_parallel_global == NULL) {
                fprintf(stderr, "[Errore] Allocazione array gather fallita.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                exit(EXIT_FAILURE);
            }
        }

        // 2. Colleziona le dimensioni
        MPI_Gather(&my_Y_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, col_comm);

        // 3. Il Master calcola gli offset
        if (rank == 0) {
            // MAGIA NERA PER L'IDE: L'asserzione lo costringe a fidarsi e azzera i warning
            assert(displs != NULL && recvcounts != NULL);

            displs[0] = 0;
            for (int r = 1; r < dims[0]; r++) {
                displs[r] = displs[r - 1] + recvcounts[r - 1];
            }
        }

        // 4. Gather finale
        MPI_Gatherv(row_Y, my_Y_size, MPI_DOUBLE,
                    (rank == 0 ? Y_parallel_global : NULL),
                    recvcounts, displs, MPI_DOUBLE, 0, col_comm);
        free(row_Y);
    }

    // 6. Validazione Seriale e Confronto
    if (rank == 0) {
        run_validation(args, Y_parallel_global, parallel_avg_time);
        free(Y_parallel_global); free(recvcounts); free(displs);
    }

    // 7. Chiusura
    free(local_A); free(local_X); free(local_Y);
    MPI_Comm_free(&row_comm); MPI_Comm_free(&col_comm); MPI_Comm_free(&cart_comm);

    if (rank == 0) printf("[Sistema] Uscita.\n");
    MPI_Finalize();
    return EXIT_SUCCESS;
}
