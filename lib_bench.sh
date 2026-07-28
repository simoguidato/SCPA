#!/bin/bash
# ==============================================================================
# lib_bench.sh — funzioni comuni per run_omp_benchmark.sh e run_cuda_benchmark.sh
# Da "sourcare", non eseguire direttamente.
# ==============================================================================
set -euo pipefail

CSV_HEADER="Mode,Backend,M,N,k,MPI_Ranks,GridRows,GridCols,OMP_Threads,CUDA_Arch,Run,ParallelTime,SequentialTime,RelativeError,GFLOPS,SpeedUp"

# init_csv <path>
init_csv() {
    echo "$CSV_HEADER" > "$1"
}

# extract_data_csv <output_testuale_del_programma>
# stampa su stdout la parte dopo "DATA_CSV:" oppure stringa vuota
extract_data_csv() {
    sed -n 's/^DATA_CSV://p' <<< "$1"
}

# append_row <csv_path> <mode> <backend> <M> <N> <k> <np> <grid_rows> <grid_cols> \
#            <omp_threads> <cuda_arch> <run> <raw_data_csv>
append_row() {
    local csv=$1 mode=$2 backend=$3 M=$4 N=$5 k=$6 np=$7 gr=$8 gc=$9
    shift 9
    local omp=$1 arch=$2 run=$3 raw=$4
    if [ -z "$raw" ]; then
        echo "[Errore] DATA_CSV mancante: $mode $backend M=$M N=$N k=$k np=$np run=$run" >&2
        return 1
    fi
    echo "$mode,$backend,$M,$N,$k,$np,$gr,$gc,$omp,$arch,$run,$raw" >> "$csv"
}

# build_cuda_kernel <kernel_id>
# compila cuda/kernel_cuda_<id>.cu + core comuni in ./app_cuda_<id>
build_cuda_kernel() {
    local id=$1
    local cuda_home="${CUDA_HOME:-/usr/local/cuda}"
    : "${CUDA_ARCH:?Impostare CUDA_ARCH, ad esempio 80 oppure 89}"

    # 'mpicc -show' stampa la riga di compilazione completa usata dal wrapper
    # (funziona sia con MPICH che con OpenMPI, a differenza di --showme:incdirs
    # che esiste solo in OpenMPI). Da lì estraiamo solo i flag -I, che servono
    # a nvcc per compilare kernel_cuda_4.cu (che include utils.h -> mpi_grid.h
    # -> mpi.h). Gli altri kernel non ne hanno bisogno, ma non fa danno passarli.
    local compile_line
    if ! compile_line=$(mpicc -show 2>/dev/null); then
        echo "[Errore] 'mpicc -show' fallito: verifica di aver caricato i moduli MPI (module load gnu/... mpich/...)." >&2
        return 1
    fi
    local mpi_inc_flags=()
    for tok in $compile_line; do
        case "$tok" in
            -I*) mpi_inc_flags+=("$tok") ;;
        esac
    done

    echo -n "Compilazione kernel CUDA ${id}... "
    nvcc -O3 -arch="sm_${CUDA_ARCH}" -Iinclude "${mpi_inc_flags[@]}" \
        -c "cuda/kernel_cuda_${id}.cu" -o "kernel_cuda_${id}.o"
    mpicc -O3 -Iinclude \
        core/main.c core/mpi_grid.c core/args.c core/utils.c core/validation.c \
        "kernel_cuda_${id}.o" -o "app_cuda_${id}" \
        -L"${cuda_home}/lib64" -lcudart -lstdc++ -lm
    echo "OK"
}

# build_omp_binary
build_omp_binary() {
    echo -n "Compilazione binario OpenMP (./SCPA)... "
    mpicc -O3 -fopenmp -Iinclude \
        core/main.c core/mpi_grid.c core/args.c core/utils.c core/validation.c \
        omp/kernel.c omp/kernel_naive.c \
        -o SCPA -lm
    echo "OK"
}