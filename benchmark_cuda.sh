#!/bin/bash

# ==============================================================================
# BENCHMARK CUDA - CONFIGURAZIONE GERARCHICA (core/, cuda/, include/)
# ==============================================================================

OUTPUT_FILE="risultati_cuda_stat.csv"
echo "Mode,M,N,k,MPI_Ranks,OMP_Threads,Run,ParallelTime,SequentialTime,RelativeError,GFLOPS,SpeedUp" > $OUTPUT_FILE

K_VALUES=(3 6 8 20 32)
SIZES=("2000 2000" "4000 4000" "8000 8000" "8000 2000" "2000 8000" "1000 4000" "4000 1000")

KERNELS=(
    "0 CUDA_Naive"
    "1 CUDA_Opt2D"
    "2 CUDA_Tiled"
    "3 CUDA_WarpRow"
)

echo "=========================================="
echo " 1. FASE DI COMPILAZIONE (Gerarchica)"
echo "=========================================="

for k_info in "${KERNELS[@]}"; do
    K_ID=$(echo $k_info | awk '{print $1}')
    MODE=$(echo $k_info | awk '{print $2}')

    echo -n "Compilazione $MODE... "

    # 1. Compiliamo l'oggetto CUDA
    # Leggiamo da cuda/ e cerchiamo gli header in include/
    nvcc -O3 -Iinclude -c "cuda/kernel_cuda_${K_ID}.cu" -o "kernel_cuda_${K_ID}.o" 2>/dev/null
    if [ $? -ne 0 ]; then echo "ERRORE NVCC (CUDA)!"; exit 1; fi

    # 2. Compiliamo i file C e linkiamo tutto
    # Leggiamo da core/ e cerchiamo gli header in include/
    # Aggiungiamo -Dcompute_local_gemm_naive=compute_local_gemm
    # Questo 'inganna' il compilatore: ogni volta che vede il nome 'naive',
    # usa la funzione CUDA standard.
    mpicc -O3 -Iinclude -Dcompute_local_gemm_naive=compute_local_gemm \
          core/main.c core/mpi_grid.c core/args.c core/utils.c core/validation.c \
          "kernel_cuda_${K_ID}.o" \
          -o "app_cuda_${K_ID}" -L/usr/local/cuda/lib64 -lcudart -lm

    if [ $? -eq 0 ]; then echo "OK"; else echo "ERRORE MPICC (C)!"; exit 1; fi
done

echo ""
echo "=========================================="
echo " 2. ESECUZIONE (5 RUN PER CONFIG)"
echo "=========================================="

for size in "${SIZES[@]}"; do
    M=$(echo $size | awk '{print $1}')
    N=$(echo $size | awk '{print $2}')
    for k in "${K_VALUES[@]}"; do
        for k_info in "${KERNELS[@]}"; do
            K_ID=$(echo $k_info | awk '{print $1}')
            MODE=$(echo $k_info | awk '{print $2}')

            # WARMUP
            mpirun -np 1 "./app_cuda_${K_ID}" $M $N $k 1 0 0 > /dev/null 2>&1

            for run in {1..5}; do
                [ "$run" -eq 1 ] && VAL=1 || VAL=0
                echo "Run $run: $MODE | ${M}x${N} | k=$k"

                OUTPUT=$(mpirun -np 1 "./app_cuda_${K_ID}" $M $N $k 1 $VAL 0)
                RAW=$(echo "$OUTPUT" | grep "^DATA_CSV:" | sed 's/^DATA_CSV://')

                if [ ! -z "$RAW" ]; then
                    echo "$MODE,$M,$N,$k,1,GPU,$run,$RAW" >> $OUTPUT_FILE
                fi
            done
        done
    done
done
echo "Fine! Dati salvati in $OUTPUT_FILE"