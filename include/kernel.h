#ifndef KERNEL_H
#define KERNEL_H


// Questa guardia assicura che extern "C" sia visto solo dai compilatori C++ (NVCC)
// ma non dai compilatori C (GCC), che non capirebbero il comando.
#ifdef __cplusplus
extern "C" {
#endif

// Qui mettiamo le funzioni che devono essere chiamate dal main.c
void setup_device_memory(int M, int N, int k, const float *A, const float *X);
void free_device_memory(int M, int N, int k, float *Y);
void compute_local_gemm(int M, int N, int k, const float *A, const float *X, float *Y);
void compute_local_gemm_naive(int M, int N, int k, const float *A, const float *X, float *Y);

#ifdef __cplusplus
}
#endif

// La macro CHECK_CUDA rimane protetta per NVCC
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CHECK_CUDA(call) { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA Error in %s at line %d: %s (%s)\n", \
__FILE__, __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
}
#endif

#endif