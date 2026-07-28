#!/bin/bash
# ==============================================================================
# run_cuda_benchmark.sh — Benchmark CUDA (tutti i kernel 0-4)
# Sostituisce benchmark_cuda.sh, cuda_mpi_np.sh, kernel_4cuda.sh,
# kernel_cuda_k_grandi.sh in un unico script parametrico.
#
# Variabili d'ambiente:
#   CUDA_ARCH       (obbligatoria)  es. 80, 86, 89
#   NUM_RUNS        (default 3)
#   MPI_PROCS_LIST  (default "1")  es. "1 2 4" per testare più GPU/rank
#   KERNEL_IDS      (default "0 1 2 3 4")
#   K_SET           (default "small") "small" -> 3 6 8 20 32, "large" -> 64 128, "all" -> entrambi
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

for id in $KERNEL_IDS; do
    build_cuda_kernel "$id"
done

for np in "${MPI_PROCS[@]}"; do
    # Griglia: per i run CUDA a singolo rank la griglia è 1x1;
    # con più rank/GPU si lascia scegliere all'utente via GRID_ROWS/GRID_COLS,
    # altrimenti automatica (0 0).
    GR="${GRID_ROWS:-0}"
    GC="${GRID_COLS:-0}"
    if [ "$np" -eq 1 ]; then GR=1; GC=1; fi

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

echo "Benchmark CUDA completato: $OUTPUT_FILE"