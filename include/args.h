#ifndef ARGS_H
#define ARGS_H

typedef struct {
    int M, N, k;
    int do_warmup;
    int do_validate;
    int kernel_type;
    int grid_rows;   // 0 = automatico (MPI_Dims_create)
    int grid_cols;   // 0 = automatico
} AppArgs;

AppArgs parse_arguments(int argc, char *argv[], int rank, int num_procs);

#endif