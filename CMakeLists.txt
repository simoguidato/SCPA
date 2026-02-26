cmake_minimum_required(VERSION 3.22)
project(SCPA C)

set(CMAKE_C_STANDARD 11)

# Definisci i file sorgente
set(SOURCES
        src/main.c
        src/mpi_grid.c
        src/utils.c
        include/utils.h
        include/kernel.h
        src/kernel.c
)

add_executable(SCPA ${SOURCES})

# 1. Specifica la cartella degli header in modo moderno
target_include_directories(SCPA PUBLIC include)

# 2. Cerca e collega MPI (Requisito punto 8)
find_package(MPI REQUIRED)
target_link_libraries(SCPA PUBLIC MPI::MPI_C)
