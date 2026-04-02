#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "args.h"

static int parse_positive_int(const char *str, int rank, const char *var_name) {
    char *endptr;
    long val = strtol(str, &endptr, 10);
    if (*endptr != '\0' || str == endptr || val <= 0) {
        if (rank == 0) fprintf(stderr, "[Errore] Input: %s non valido.\n", var_name);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
    return (int)val;
}

static int parse_boolean_flag(const char *str, int rank) {
    char *endptr;
    long val = strtol(str, &endptr, 10);
    if (*endptr != '\0' || str == endptr || (val != 0 && val != 1)) {
        if (rank == 0) fprintf(stderr, "[Errore] Input: '%s' non valido (richiesto 0 o 1).\n", str);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
    }
    return (int)val;
}

// Sostituisci la funzione parse_arguments con questa:
AppArgs parse_arguments(int argc, char *argv[], int rank, int num_procs) {
    AppArgs args;
    // Ora accettiamo fino a 7 argomenti!
    if (argc < 4 || argc > 7) {
        if (rank == 0) printf("Uso: mpirun -np <P> %s <M> <N> <k> [warmup] [validate] [kernel_type]\n", argv[0]);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    args.M = parse_positive_int(argv[1], rank, "M");
    args.N = parse_positive_int(argv[2], rank, "N");
    args.k = parse_positive_int(argv[3], rank, "k");

    args.do_warmup = (argc >= 5) ? parse_boolean_flag(argv[4], rank) : 1;
    args.do_validate = (argc >= 6) ? parse_boolean_flag(argv[5], rank) : 1;
    args.kernel_type = (argc == 7) ? parse_boolean_flag(argv[6], rank) : 0; // Se non passato, usa 0 di default

    return args;
}