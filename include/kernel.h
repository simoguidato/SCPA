#ifndef KERNEL_H
#define KERNEL_H

// Calcola Y_local = A_local * X_local
// Questa firma va bene sia per OpenMP che per CUDA
void compute_local_gemm(int local_M, int local_N, int k, 
                        const double *A_local, const double *X_local, 
                        double *Y_local);
void setup_device_memory(int M, int N, int k, const double *A, const double *X);
void free_device_memory(int M, int N, int k, double *Y);

#endif