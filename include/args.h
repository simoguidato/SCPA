#ifndef ARGS_H
#define ARGS_H

typedef struct {
    int M, N, k;
    int do_warmup;
} AppArgs;

AppArgs parse_arguments(int argc, char *argv[], int rank, int num_procs);

#endif