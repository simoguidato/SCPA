#!/bin/bash

# ==============================================================================
# SCRIPT UNIFICATO: COMPILAZIONE + BENCHMARK OVERSUBSCRIPTION (TUTTI I KERNEL)
# ==============================================================================

OUTPUT_FILE="risultati_cuda_mpi_all.csv"

echo "=========================================="
echo " 1. CARICAMENTO MODULI AMBIENTE"
echo "=========================================="
module load cuda/12.8
if [ $? -ne 0 ]; then
    echo "[ERRORE] Impossibile caricare il modulo CUDA."
    exit 1
fi
echo "Modulo CUDA 12.8 caricato con successo."

# Array delle configurazioni
KERNELS=(
    "0 CUDA_Naive"
    "1 CUDA_Opt2D"
    "2 CUDA_Tiled"
    "3 CUDA_WarpRow"
)

K_VALUES=(3 6 8 20 32)
SIZES=("2000 2000" "4000 4000" "8000 8000" "8000 2000" "2000 8000" "1000 4000" "4000 1000")
MPI_PROCS=(1 2 4 8 16)

echo ""
echo "=========================================="
echo " 2. FASE DI COMPILAZIONE (Tutti i Kernel)"
echo "=========================================="
for k_info in "${KERNELS[@]}"; do
    K_ID=$(echo $k_info | awk '{print $1}')
    MODE=$(echo $k_info | awk '{print $2}')
    EXE_NAME="app_cuda_${K_ID}"

    echo -n "Compilazione $MODE... "
    nvcc -O3 -arch=native -Iinclude -c "cuda/kernel_cuda_${K_ID}.cu" -o "kernel_cuda_${K_ID}.o"
    if [ $? -ne 0 ]; then echo "ERRORE NVCC!"; exit 1; fi

    mpicc -O3 -Iinclude -Dcompute_local_gemm_naive=compute_local_gemm \
          core/main.c core/mpi_grid.c core/args.c core/utils.c core/validation.c \
          "kernel_cuda_${K_ID}.o" \
          -o "$EXE_NAME" -L/usr/local/cuda/lib64 -lcudart -lm
    if [ $? -ne 0 ]; then echo "ERRORE MPICC!"; exit 1; fi
    echo "OK"
done

# 3. ESECUZIONE BENCHMARK
echo ""
echo "=========================================="
echo " 3. TEST MPI + CUDA (OVERSUBSCRIPTION)"
echo " GPU: 1x NVIDIA Quadro RTX 5000"
echo "=========================================="

echo "Mode,M,N,k,MPI_Ranks,OMP_Threads,Run,ParallelTime,SequentialTime,RelativeError" > $OUTPUT_FILE

for np in "${MPI_PROCS[@]}"; do
    # Binding per i processi CPU
    BIND_FLAGS="--bind-to socket --map-by socket"
    if [ "$np" -eq 1 ]; then BIND_FLAGS=""; fi

    for size in "${SIZES[@]}"; do
        M=$(echo $size | awk '{print $1}')
        N=$(echo $size | awk '{print $2}')
        for k in "${K_VALUES[@]}"; do
            for k_info in "${KERNELS[@]}"; do
                K_ID=$(echo $k_info | awk '{print $1}')
                MODE=$(echo $k_info | awk '{print $2}')
                EXE_NAME="app_cuda_${K_ID}"

                echo "------------------------------------------------"
                echo "Config: NP=$np | Matrice: ${M}x${N} | Kernel: $MODE"

                # WARMUP
                mpirun -np $np $BIND_FLAGS "./$EXE_NAME" $M $N $k 1 0 0 > /dev/null 2>&1

                for run in {1..3}; do
                    [ "$run" -eq 1 ] && VAL=1 || VAL=0
                    OUTPUT=$(mpirun -np $np $BIND_FLAGS "./$EXE_NAME" $M $N $k 1 $VAL 0)
                    RAW=$(echo "$OUTPUT" | grep "^DATA_CSV:" | sed 's/^DATA_CSV://')

                    if [ ! -z "$RAW" ]; then
                        echo "$MODE,$M,$N,$k,$np,GPU,$run,$RAW" >> $OUTPUT_FILE
                        echo "  > Run $run: OK"
                    else
                        echo "  > Run $run: ERRORE"
                    fi
                done
            done
        done
    done
done

echo "=========================================="
echo "Benchmark completato! Dati salvati in $OUTPUT_FILE"