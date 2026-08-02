#ifndef CUDA_DEVICE_UTILS_CUH
#define CUDA_DEVICE_UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// I launcher MPI più comuni espongono il rank locale del nodo con una di
// queste variabili. CUDA_VISIBLE_DEVICES può inoltre restringere la vista
// delle GPU disponibili al processo.
static inline int cuda_local_rank(void) {
    const char *value = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (value == NULL) value = getenv("PMI_LOCAL_RANK");
    if (value == NULL) value = getenv("SLURM_LOCALID");
    return value == NULL ? 0 : atoi(value);
}

static inline void cuda_select_device_or_die(void) {
    int num_devices = 0;
    cudaError_t status = cudaGetDeviceCount(&num_devices);
    if (status != cudaSuccess || num_devices <= 0) {
        fprintf(stderr, "CUDA: nessuna GPU disponibile: %s\n",
                cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }

    status = cudaSetDevice(cuda_local_rank() % num_devices);
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA: selezione GPU fallita: %s\n",
                cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

#endif
