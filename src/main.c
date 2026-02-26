#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mpi_grid.h"
#include "utils.h"
#include "kernel.h"

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

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int k = atoi(argv[3]);

    // 3. Creazione Topologia Cartesiana 2D
    int dims[2] = {0, 0};
    // Lasciamo che MPI scelga la griglia migliore (es. 4 processi -> 2x2)
    MPI_Dims_create(num_procs, 2, dims);

    int periods[2] = {0, 0}; // Nessun avvolgimento (toroide)
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    // Coordinate del mio processo (my_row, my_col)
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

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

    // Pulizia e chiusura
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}