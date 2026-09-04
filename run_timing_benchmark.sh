#!/bin/bash
# ==============================================================================
# run_timing_benchmark.sh —
# Misura, a parte rispetto al benchmark principale, il tempo di:
#   - distribuzione MPI del multivettore X (MPI_Bcast)
#   - trasferimento H2D + D2H (solo per kernel CUDA; ~0 per OMP)
#
# Campagna volutamente ridotta:
# 2 kernel CUDA rappresentativi + 1 kernel OMP rappresentativo,
# tutte le taglie, tutti i k, NP=1 (isola il fenomeno dalla complessità di
# griglia).
#
# Uso:
#   CUDA_ARCH=75 NUM_RUNS=3 ./run_timing_benchmark.sh
#
# Variabili d'ambiente:
#   CUDA_ARCH        obbligatoria per la parte CUDA
#   NUM_RUNS         default 3
#   CUDA_KERNEL_IDS  default "0 2" (Naive + Tiled: il più povero e il campione)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./lib_bench.sh

OUTPUT_FILE="results/risultati_tempi.csv"
NUM_RUNS="${NUM_RUNS:-3}"
CUDA_KERNEL_IDS="${CUDA_KERNEL_IDS:-0 2}"
K_VALUES="3 6 8 20 32 64 128"
SIZES=("2000 2000" "2000 4000" "4000 2000" "6000 2000" "4000 4000" "2000 6000")

declare -A KERNEL_NAMES=(
    [0]="CUDA_Naive" [1]="CUDA_Opt2D" [2]="CUDA_Tiled"
    [3]="CUDA_WarpRow" [4]="CUDA_Transposed"
)

mkdir -p results
init_timing_csv "$OUTPUT_FILE"

if [ -n "${CUDA_ARCH:-}" ]; then
    for id in $CUDA_KERNEL_IDS; do
        build_cuda_kernel "$id"
    done
    for size in "${SIZES[@]}"; do
        read -r M N <<< "$size"
        for k in $K_VALUES; do
            for id in $CUDA_KERNEL_IDS; do
                mode="${KERNEL_NAMES[$id]}"
                for ((run = 1; run <= NUM_RUNS; ++run)); do
                    validate=$([ "$run" -eq 1 ] && echo 1 || echo 0)
                    echo "TIMING | $mode | ${M}x${N} | k=$k | run $run"
                    output=$(mpirun -np 1 "./app_cuda_${id}" "$M" "$N" "$k" 1 "$validate" 0 1 1)
                    raw=$(extract_data_csv "$output")
                    extra=$(extract_extra_times "$output")
                    append_timing_row "$OUTPUT_FILE" "$mode" "CUDA" "$M" "$N" "$k" 1 "$run" "$raw" "$extra" || true
                done
            done
        done
    done
else
    echo "CUDA_ARCH non impostata: salto la parte CUDA (solo OMP)."
fi

# --- Parte OMP (un solo kernel rappresentativo: Opt_SMP) ---
build_omp_binary
for size in "${SIZES[@]}"; do
    read -r M N <<< "$size"
    for k in $K_VALUES; do
        for ((run = 1; run <= NUM_RUNS; ++run)); do
            validate=$([ "$run" -eq 1 ] && echo 1 || echo 0)
            export OMP_NUM_THREADS="${NPROC_NODE:-$(nproc)}"
            echo "TIMING | Opt_SMP | ${M}x${N} | k=$k | run $run"
            output=$(mpirun -np 1 ./SCPA "$M" "$N" "$k" 1 "$validate" 0 1 1)
            raw=$(extract_data_csv "$output")
            extra=$(extract_extra_times "$output")
            append_timing_row "$OUTPUT_FILE" "Opt_SMP" "OMP" "$M" "$N" "$k" 1 "$run" "$raw" "$extra" || true
        done
    done
done

echo "Mini-campagna tempi completata: $OUTPUT_FILE"
