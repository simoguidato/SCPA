#include "mpi_grid.h"
#include <mpi.h>
#include <stdio.h>

void compute_grid_partition(int M, int N, int rows_grid, int cols_grid, int my_row, int my_col, GridInfo *info) {
    info->global_M = M;
    info->global_N = N;

    // --- Calcolo Righe (M) bilanciato ---
    int base_M = M / rows_grid;
    int rest_M = M % rows_grid;
    // Se la mia riga è minore del resto, prendo +1. Altrimenti base.
    info->local_M = base_M + (my_row < rest_M ? 1 : 0);
    // Formula magica per l'offset senza ricalcolare tutto
    info->offset_M = my_row * base_M + (my_row < rest_M ? my_row : rest_M);

    // --- Calcolo Colonne (N) bilanciato ---
    int base_N = N / cols_grid;
    int rest_N = N % cols_grid;
    info->local_N = base_N + (my_col < rest_N ? 1 : 0);
    info->offset_N = my_col * base_N + (my_col < rest_N ? my_col : rest_N);
    //printf("calcolo suddivisione partizioni %d(M), %d(N)",info->local_M, info->local_N);
}
void create_sub_communicators(MPI_Comm cart_comm, MPI_Comm *row_comm, MPI_Comm *col_comm) {
    int remain_dims[2];

    // Crea comunicatore per le RIGHE (varia la colonna, la riga è fissa)
    remain_dims[0] = 0; // Riga fissa
    remain_dims[1] = 1; // Colonna varia
    MPI_Cart_sub(cart_comm, remain_dims, row_comm);

    // Crea comunicatore per le COLONNE (varia la riga, la colonna è fissa)
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(cart_comm, remain_dims, col_comm);
}