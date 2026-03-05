#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpi_grid.h"
#include "utils.h"
#include "kernel.h"

int parse_positive_int(const char *str, int rank, const char *var_name) {
    char *endptr;
    long val = strtol(str, &endptr, 10);
    if (*endptr != '\0' || str == endptr || val <= 0) {
        if (rank == 0) fprintf(stderr, "[Errore] Input: %s non valido.\n", var_name);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
    return (int)val;
}

int main(int argc, char *argv[]) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc != 4) {
        if (rank == 0) printf("Uso: mpirun -np <P> %s <M> <N> <k>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int M = parse_positive_int(argv[1], rank, "M");
    int N = parse_positive_int(argv[2], rank, "N");
    int k = parse_positive_int(argv[3], rank, "k");

    if (rank == 0) printf("\n[Sistema] Avvio SPMM Parallelo: Matrice %dx%d, k=%d, Processi MPI=%d\n", M, N, k, num_procs);

    int dims[2] = {0, 0};
    MPI_Dims_create(num_procs, 2, dims);

    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    if (M < dims[0] || N < dims[1]) {
        if (rank == 0) fprintf(stderr, "[Errore] Matrice troppo piccola per la griglia.\n");
        MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    GridInfo info;
    compute_grid_partition(M, N, dims[0], dims[1], coords[0], coords[1], &info);

    MPI_Comm row_comm, col_comm;
    create_sub_communicators(cart_comm, &row_comm, &col_comm);

    if (rank == 0) printf("[MPI] Topologia Cartesiana %dx%d creata con successo.\n", dims[0], dims[1]);

    // --- 5. ALLOCAZIONE E SETUP LOCALE ---
    double *local_A = allocate_matrix(info.local_M, info.local_N);
    double *local_X = allocate_matrix(info.local_N, k);
    double *local_Y = allocate_matrix(info.local_M, k);

    if (local_A == NULL || local_X == NULL || local_Y == NULL) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }

    generate_data_locally(local_A, info, 42);

    if (coords[1] == 0) {
        for (int i = 0; i < info.local_N * k; i++) local_X[i] = 1.0;
    }
    for (int i = 0; i < info.local_M * k; i++) local_Y[i] = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("[MPI] Generazione dati locale completata. Avvio Broadcast di X...\n");

    // --- 6. COMUNICAZIONE IN (BROADCAST) ---
    MPI_Bcast(local_X, info.local_N * k, MPI_DOUBLE, 0, row_comm);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("[OpenMP] Inizio calcolo kernel distribuito...\n");

    // --- 7. CALCOLO PARALLELO ---
    compute_local_gemm(info.local_M, info.local_N, k, local_A, local_X, local_Y);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("[MPI] Calcolo completato. Avvio raccolta risultati (Reduce + Gather)...\n");

    // --- 8. RACCOLTA DATI (REDUCE + GATHER) ---
    double *row_Y = NULL;
    if (coords[1] == 0) {
        row_Y = allocate_matrix(info.local_M, k);
    }

    MPI_Reduce(local_Y, row_Y, info.local_M * k, MPI_DOUBLE, MPI_SUM, 0, row_comm);

    double *Y_parallel_global = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        Y_parallel_global = allocate_matrix(M, k);
        recvcounts = malloc(dims[0] * sizeof(int));
        displs = malloc(dims[0] * sizeof(int));
        int offset = 0;
        for (int r = 0; r < dims[0]; r++) {
            int r_local_M = M / dims[0];
            if (r == dims[0] - 1) r_local_M += M % dims[0];
            recvcounts[r] = r_local_M * k;
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    if (coords[1] == 0) {
        MPI_Gatherv(row_Y, info.local_M * k, MPI_DOUBLE,
                    Y_parallel_global, recvcounts, displs, MPI_DOUBLE,
                    0, col_comm);
        free(row_Y);
    }

    // --- 9. VERIFICA SERIALE ---
    if (rank == 0 && M <= 2000 && N <= 2000) {
        printf("[Test] Avvio verifica seriale globale...\n");
        double *A_global = allocate_matrix(M, N);
        double *X_global = allocate_matrix(N, k);
        double *Y_serial = allocate_matrix(M, k);

        if (A_global && X_global && Y_serial) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    const unsigned int seed = 42;
                    long long global_idx = (long long)i * N + j;
                    A_global[i * N + j] = (double)((global_idx + seed) % 100) / 7.0;
                }
            }
            for (int i = 0; i < N * k; i++) X_global[i] = 1.0;

            compute_serial_gemm(M, N, k, A_global, X_global, Y_serial);

            if (verify_result(Y_parallel_global, Y_serial, M * k)) {
                printf("[SUCCESSO] Risultato Parallelo = Risultato Seriale!\n");
            } else {
                printf("[ERRORE] I risultati non coincidono.\n");
            }

            free(A_global);
            free(X_global);
            free(Y_serial);
        }
        free(Y_parallel_global);
        free(recvcounts);
        free(displs);
    }

    // --- 10. CHIUSURA ---
    free(local_A);
    free(local_X);
    free(local_Y);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    if (rank == 0) printf("[Sistema] Deallocazione completata. Uscita.\n\n");
    MPI_Finalize();
    return EXIT_SUCCESS;
}