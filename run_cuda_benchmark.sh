#!/bin/bash
# ==============================================================================
# run_cuda_benchmark.sh — Benchmark CUDA (Scaling + Topologia Griglia)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./lib_bench.sh

: "${CUDA_ARCH:?Impostare CUDA_ARCH, ad esempio 80}"
OUTPUT_FILE="${OUTPUT_FILE:-results/risultati_cuda.csv}"
NUM_RUNS="${NUM_RUNS:-3}"
MPI_PROCS_LIST="${MPI_PROCS_LIST:-1 2 4 8}"
KERNEL_IDS="${KERNEL_IDS:-0 1 2 3 4}"
K_SET="${K_SET:-small}"
TEST_TOPOLOGY="${TEST_TOPOLOGY:-1}" # Se 1, testa tutte le combinazioni di griglia per ogni NP

declare -A KERNEL_NAMES=(
    [0]="CUDA_Naive" [1]="CUDA_Opt2D" [2]="CUDA_Tiled"
    [3]="CUDA_WarpRow" [4]="CUDA_Transposed"
)

case "$K_SET" in
    small) K_VALUES="3 6 8 20 32" ;;
    large) K_VALUES="64 128" ;;
    all)   K_VALUES="3 6 8 20 32 64 128" ;;
    *) echo "K_SET non valido: $K_SET (usa small|large|all)" >&2; exit 1 ;;
esac

SIZES=("2000 2000" "2000 4000" "4000 2000" "6000 2000" "4000 4000" "2000 6000")
read -r -a MPI_PROCS <<< "$MPI_PROCS_LIST"

mkdir -p "$(dirname "$OUTPUT_FILE")"
init_csv "$OUTPUT_FILE"

# Compilazione dei kernel
for id in $KERNEL_IDS; do
    build_cuda_kernel "$id"
done

# Esecuzione
for np in "${MPI_PROCS[@]}"; do
    SHAPES=()
    if [ "$np" -eq 1 ]; then
        SHAPES=("1 1")
    elif [ "$TEST_TOPOLOGY" -eq 1 ]; then
        # Innesto topologia: Trova tutte le combinazioni PxQ = np
        for ((p = 1; p <= np; p++)); do
            if [ $((np % p)) -eq 0 ]; then
                q=$((np / p))
                SHAPES+=("$p $q")
            fi
        done
    else
        # Se TEST_TOPOLOGY=0, usa 0 0 e lascia fare a MPI_Dims_create nel codice C
        SHAPES=("0 0")
    fi

    for shape in "${SHAPES[@]}"; do
        read -r GR GC <<< "$shape"
        echo "=========================================================="
        if [ "$GR" -eq 0 ]; then
            echo "Esecuzione con np=$np (Griglia Automatica MPI_Dims_create)"
        else
            echo "Esecuzione con np=$np (Griglia Forzata: ${GR}x${GC})"
        fi
        echo "=========================================================="

        for size in "${SIZES[@]}"; do
            read -r M N <<< "$size"
            for k in $K_VALUES; do
                for id in $KERNEL_IDS; do
                    mode="${KERNEL_NAMES[$id]}"
                    for ((run = 1; run <= NUM_RUNS; ++run)); do
                        validate=$([ "$run" -eq 1 ] && echo 1 || echo 0)
                        echo "CUDA | $mode | ${M}x${N} | k=$k | np=$np grid=${GR}x${GC} | run $run"
                        output=$(mpirun -np "$np" "./app_cuda_${id}" \
                            "$M" "$N" "$k" 1 "$validate" 0 "$GR" "$GC")
                        raw=$(extract_data_csv "$output")
                        append_row "$OUTPUT_FILE" "$mode" "CUDA" "$M" "$N" "$k" "$np" "$GR" "$GC" \
                            "" "$CUDA_ARCH" "$run" "$raw" || true
                    done
                done
            done
        done
    done
done

echo "Benchmark CUDA completato con successo: $OUTPUT_FILE"