#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpi_grid.h"
#include "utils.h"
#include "kernel.h"

int parse_positive_int(const char *str, int rank, const char *var_name) {
    char *endptr;
    long val = strtol(str, &endptr, 10);

    // Controlla se la stringa conteneva testo, se è vuota o se il valore è <= 0
    if (*endptr != '\0' || str == endptr || val <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Errore di input: '%s' non e' un valore valido per %s (richiesto intero > 0).\n", str, var_name);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE); // Spegne i warning di Clang-Tidy
    }
    return (int)val;
}

int main(int argc, char *argv[]) {
    // 1. Inizializzazione MPI
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 2. Lettura argomenti (M, N, k) richiesti dal progetto
    if (argc != 4) {
        if (rank == 0) printf("Uso: mpirun -np <P> %s <M> <N> <k>\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int M = parse_positive_int(argv[1], rank, "M");
    int N = parse_positive_int(argv[2], rank, "N");
    int k = parse_positive_int(argv[3], rank, "k");

    // 3. Creazione Topologia Cartesiana 2D
    int dims[2] = {0, 0};
    // Lasciamo che MPI scelga la griglia migliore (es. 4 processi -> 2x2)
    MPI_Dims_create(num_procs, 2, dims);

    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    // Coordinate del mio processo (my_row, my_col)
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    if (M < dims[0] || N < dims[1]) {
        if (rank == 0) {
            fprintf(stderr, "Errore: Matrice %dx%d troppo piccola per una griglia %dx%d.\n",
                    M, N, dims[0], dims[1]);
        }
        MPI_Comm_free(&cart_comm);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    // 4. Calcolo partizione e creazione sotto-comunicatori
    GridInfo info;
    compute_grid_partition(M, N, dims[0], dims[1], coords[0], coords[1], &info);

    MPI_Comm row_comm, col_comm;
    create_sub_communicators(cart_comm, &row_comm, &col_comm);

    // --- FINE SETUP ---
    // Test di debug stampato in ordine
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d (Riga %d, Col %d): A_local [%d x %d]\n",
            rank, coords[0], coords[1], info.local_M, info.local_N);

    // --- 5. ALLOCAZIONE MEMORIA LOCALE ---
    // Usiamo la tua funzione con posix_memalign per le massime prestazioni
    double *local_A = allocate_matrix(info.local_M, info.local_N);
    double *local_X = allocate_matrix(info.local_N, k);
    double *local_Y = allocate_matrix(info.local_M, k);

    // Controllo di sicurezza
    if (local_A == NULL || local_X == NULL || local_Y == NULL) {
        fprintf(stderr, "Errore di allocazione sul processo %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }

    // --- 6. PREPROCESSING: GENERAZIONE DATI ---
    // Genera la sottomatrice A coerentemente con gli indici globali
    generate_data_locally(local_A, info, 42);

    // Per testare il kernel prima di fare comunicazioni complesse,
    // riempiamo X di 1.0 e azzeriamo Y.
    for (int i = 0; i < info.local_N * k; i++) {
        local_X[i] = 1.0;
    }
    for (int i = 0; i < info.local_M * k; i++) {
        local_Y[i] = 0.0;
    }

    // Sincronizziamo tutti i processi prima di far partire il calcolo
    MPI_Barrier(MPI_COMM_WORLD);

    // --- 7. ESECUZIONE DEL NUCLEO DI CALCOLO (KERNEL) ---
    if (rank == 0) printf("Inizio calcolo locale OpenMP...\n");

    // Richiama il kernel OpenMP che calcola Y = A * X
    compute_local_gemm(info.local_M, info.local_N, k, local_A, local_X, local_Y);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Calcolo locale completato!\n");
//  -----7.5 calcolo seriale e confronto
    if (rank == 0 && M <= 2000 && N <= 2000) {
        printf("--- AVVIO VERIFICA SERIALE GLOBALE ---\n");

        // Alloca memoria per le matrici globali
        double *A_global = allocate_matrix(M, N);
        double *X_global = allocate_matrix(N, k);
        double *Y_serial = allocate_matrix(M, k);

        if (A_global != NULL && X_global != NULL && Y_serial != NULL) {

            // TODO: 1. Generare l'intera A_global coerentemente con le sottomatrici.
            // TODO: 2. Generare X_global con 1.0 (per rispecchiare il setup di test).

            // Esecuzione del calcolo seriale
            compute_serial_gemm(M, N, k, A_global, X_global, Y_serial);
            printf("Calcolo seriale completato.\n");

            // TODO: 3. Raccogliere tutti i local_Y in un Y_global usando MPI_Gather/Reduce.
            // TODO: 4. Chiamare verify_result(Y_global, Y_serial, M * k).

            free(A_global);
            free(X_global);
            free(Y_serial);
        } else {
            fprintf(stderr, "Rank 0: Impossibile allocare memoria per il test seriale.\n");
        }
    }

    // --- 8. DEALLOCAZIONE MEMORIA ---
    free(local_A);
    free(local_X);
    free(local_Y);

    // Pulizia e chiusura
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}