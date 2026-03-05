cmake_minimum_required(VERSION 3.22)
project(SCPA C)

set(CMAKE_C_STANDARD 11)

# 1. Flag di Ottimizzazione per HPC (Fondamentale per i GFLOPS!)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# Definisci i file sorgente e gli header (utile per visualizzarli in CLion)
set(SOURCES
        src/main.c
        src/mpi_grid.c
        src/utils.c
        src/kernel.c
        include/utils.h
        include/kernel.h
        include/mpi_grid.h
)

add_executable(SCPA ${SOURCES})

# Specifica la cartella degli header
target_include_directories(SCPA PUBLIC include)

# 2. Cerca MPI (Requisito del progetto)
find_package(MPI REQUIRED)

# 3. Cerca OpenMP (Per il kernel di calcolo)
find_package(OpenMP REQUIRED)

# 4. Collega tutte le librerie all'eseguibile (MPI, OpenMP e Math 'm')
target_link_libraries(SCPA
        PUBLIC
        MPI::MPI_C
        OpenMP::OpenMP_C
        m
)
