#!/bin/bash

# ==============================================================================
# SCRIPT DI BENCHMARK HPC - MPI + OpenMP (Analisi Statistica)
# ==============================================================================

EXE="./SCPA"
OUTPUT_FILE="risultati_avanzati.csv"

# Intestazione completa
echo "Mode,M,N,k,MPI_Ranks,OMP_Threads,Run,ParallelTime,SequentialTime,RelativeError,GFLOPS,SpeedUp" > $OUTPUT_FILE

CONFIGS=(
    "Serial 1 1 0"
    "Naive_SMP 1 40 1"
    "Naive_Hybrid 2 20 1"
    "Opt_SMP 1 40 0"         # NP:1 (Memory Bound)
    "Opt_Hybrid 2 20 0"      # NP:2 (Sweet spot, NUMA sbloccato)
    "Opt_Hybrid 4 10 0"      # NP:4 (Inizio overhead)
    "Opt_Hybrid 8 5 0"       # NP:8 (Crollo di comunicazione)
)

K_VALUES=(3 6 8 20 32)
SIZES=("2000 2000" "4000 4000" "8000 2000" "2000 8000")

export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "Avvio test statistico (5 Run per configurazione)..."

for size in "${SIZES[@]}"; do
    M=$(echo $size | awk '{print $1}')
    N=$(echo $size | awk '{print $2}')
    for k in "${K_VALUES[@]}"; do
        for config in "${CONFIGS[@]}"; do
            MODE=$(echo $config | awk '{print $1}')
            NP=$(echo $config | awk '{print $2}')
            OMP=$(echo $config | awk '{print $3}')
            KTYPE=$(echo $config | awk '{print $4}')
            export OMP_NUM_THREADS=$OMP

            # GESTIONE BINDING CONDIZIONALE (Solo se NP > 1)
            if [ "$NP" -gt 1 ]; then
                BIND_FLAGS="--bind-to socket --map-by socket"
            else
                BIND_FLAGS=""
            fi
            mpirun -np $NP $BIND_FLAGS $EXE $M $N $k 1 0 $KERNEL > /dev/null 2>&1
            # Ripetizioni multiple per media e varianza
            for run in {1..5}; do
                            # Esegui la validazione pesante SOLO alla run 1
                            if [ "$run" -eq 1 ]; then
                                VALIDATE=1
                            else
                                VALIDATE=0
                            fi

                            echo "Esecuzione: $MODE | Matrice: ${M}x${N} | k: $k | Run: $run (Val: $VALIDATE)"

                            # Passiamo 1 (warmup attivo) e $VALIDATE come argomenti finali
                            OUTPUT=$(mpirun -np $NP $BIND_FLAGS $EXE $M $N $k 1 $VALIDATE $KTYPE)

                            # Parser a prova di crash (gestisce "e-", "nan", "inf")
                            RAW_DATA=$(echo "$OUTPUT" | grep "^DATA_CSV:" | sed 's/^DATA_CSV://')

                            if [ ! -z "$RAW_DATA" ]; then
                                echo "$MODE,$M,$N,$k,$NP,$OMP,$run,$RAW_DATA" >> $OUTPUT_FILE
                            else
                                echo "[Errore] Dati non trovati per questa run."
                            fi
            done
        done
    done
done

echo "Benchmark terminato. Dati salvati in $OUTPUT_FILE"