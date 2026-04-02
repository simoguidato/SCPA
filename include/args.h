#ifndef ARGS_H
#define ARGS_H

typedef struct {
    int M, N, k;
    int do_warmup;
    int do_validate;
    int kernel_type;
} AppArgs;

AppArgs parse_arguments(int argc, char *argv[], int rank, int num_procs);

#endif