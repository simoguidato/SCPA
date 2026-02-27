#ifndef UTILS_H
#define UTILS_H

#include "mpi_grid.h"

// Alloca memoria allineata per performance migliori (OpenMP)
double* allocate_matrix(int rows, int cols);

// Genera i dati locali in modo che siano coerenti con la versione seriale
void generate_data_locally(double *local_A, GridInfo info, unsigned int seed);

// Funzione per confrontare il risultato parallelo con quello seriale
int verify_result(double *Y_parallel, double *Y_serial, int size);

void compute_serial_gemm(int M, int N, int k, const double *A, const double *X, double *Y);

#endif