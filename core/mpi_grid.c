#include "mpi_grid.h"
#include <mpi.h>

void compute_grid_partition(int M, int N, int rows_grid, int cols_grid, int my_row, int my_col, GridInfo *info) {
    info->global_M = M;
    info->global_N = N;

    // Divisione base (arrotondata per difetto)
    info->local_M = M / rows_grid;
    info->local_N = N / cols_grid;

    // Gestione del resto: l'ultimo processo di ogni riga/colonna riceve gli elementi extra 
    if (my_row == rows_grid - 1) info->local_M += M % rows_grid;
    if (my_col == cols_grid - 1) info->local_N += N % cols_grid;

    // Calcolo del punto di partenza (offset) nella matrice globale
    info->offset_M = my_row * (M / rows_grid);
    info->offset_N = my_col * (N / cols_grid);
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