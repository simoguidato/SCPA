#!/bin/bash
# ==============================================================================
# run_omp_benchmark.sh — Benchmark MPI + OpenMP
# Sostituisce benchmark.sh e openmp_kgrandi.sh in un unico script parametrico.
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"
source ./lib_bench.sh

OUTPUT_FILE="${OUTPUT_FILE:-results/risultati_omp.csv}"
NUM_RUNS="${NUM_RUNS:-3}"
NPROC_NODE="${NPROC_NODE:-$(nproc)}"

mkdir -p "$(dirname "$OUTPUT_FILE")"
init_csv "$OUTPUT_FILE"

# Config: "Nome NP OMP KernelType[0=Opt,1=Naive] GridRows GridCols"
# GridRows/GridCols=0 0 => griglia automatica (MPI_Dims_create)
CONFIGS=(
    "Serial        1  1              0 0 0"
    "Naive_SMP     1  ${NPROC_NODE}  1 0 0"
    "Naive_Hybrid  2  $((NPROC_NODE/2)) 1 0 0"
    "Opt_SMP       1  ${NPROC_NODE}  0 0 0"
    "Opt_Hybrid    2  $((NPROC_NODE/2)) 0 0 0"
    "Opt_Hybrid    4  $((NPROC_NODE/4)) 0 0 0"
    "Opt_Hybrid    8  $((NPROC_NODE/8)) 0 0 0"
    # Stesso -np=8, griglie diverse: verifica esplicita del requisito
    # "griglia con estensione scelta dall'utente"
    "Opt_Grid1x8   8  1              0 1 8"
    "Opt_Grid8x1   8  1              0 8 1"
    "Opt_Grid2x4   8  1              0 2 4"
)

K_SETS=("small:3 6 8 20 32" "large:64 128")
SIZES=("2000 2000" "2000 4000" "4000 2000" "6000 2000" "4000 4000" "2000 6000")

export OMP_PLACES=cores
export OMP_PROC_BIND=close

build_omp_binary

for k_set in "${K_SETS[@]}"; do
    K_VALUES="${k_set#*:}"
    for size in "${SIZES[@]}"; do
        read -r M N <<< "$size"
        for k in $K_VALUES; do
            for config in "${CONFIGS[@]}"; do
                read -r MODE NP OMP KTYPE GR GC <<< "$config"
                export OMP_NUM_THREADS=$OMP

                # Sintassi Hydra (MPICH): niente --map-by (è OpenMPI-only)
                BIND_FLAGS=()
                if [ "$NP" -gt 1 ]; then
                    BIND_FLAGS=(-bind-to socket)
                fi

                for ((run = 1; run <= NUM_RUNS; ++run)); do
                    VALIDATE=$([ "$run" -eq 1 ] && echo 1 || echo 0)
                    echo "OMP | $MODE | ${M}x${N} | k=$k | np=$NP grid=${GR}x${GC} | run $run"
                    OUTPUT=$(mpirun -np "$NP" "${BIND_FLAGS[@]}" ./SCPA \
                        "$M" "$N" "$k" 1 "$VALIDATE" "$KTYPE" "$GR" "$GC")
                    RAW=$(extract_data_csv "$OUTPUT")
                    append_row "$OUTPUT_FILE" "$MODE" "OMP" "$M" "$N" "$k" "$NP" "$GR" "$GC" \
                        "$OMP" "" "$run" "$RAW" || true
                done
            done
        done
    done
done

echo "Benchmark OMP completato: $OUTPUT_FILE"