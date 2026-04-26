#ifndef UTILS_H
#define UTILS_H

#include "mpi_grid.h"

// Alloca memoria allineata per performance migliori (OpenMP)
float *allocate_matrix(int rows, int cols);

// Genera i dati locali in modo che siano coerenti con la versione seriale
void generate_data_locally(float *local_A, GridInfo info, unsigned int seed);

// Funzione per confrontare il risultato parallelo con quello seriale
int verify_result(const float *Y_parallel, const float *Y_serial, int size);

void compute_serial_gemm(int M, int N, int k, const float *A, const float *X, float *Y);
void generate_X_locally(float *local_X, int local_N, int k, int offset_N, unsigned int seed);
#endif