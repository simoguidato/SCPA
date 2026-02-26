#ifndef MPI_GRID_H
#define MPI_GRID_H

// Struttura per gestire la distribuzione della matrice A sulla griglia 2D
typedef struct {
    int global_M, global_N;
    int local_M, local_N;
    int offset_M, offset_N; // Coordinate di inizio del blocco locale nella matrice globale
} GridInfo;

// Calcola come partizionare la matrice globale tra i processi
void compute_grid_partition(int M, int N, int rows_grid, int cols_grid, int my_row, int my_col, GridInfo *info);

#endif