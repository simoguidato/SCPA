#ifndef VALIDATION_H
#define VALIDATION_H
#include "args.h"

void run_validation(AppArgs args,int rank, const double *Y_parallel_global, double parallel_avg_time);

#endif