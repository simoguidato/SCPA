cmake_minimum_required(VERSION 3.22)
project(SCPA LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_CUDA "Abilita l'accelerazione GPU con CUDA" OFF)

# Ora puntiamo tutto alla cartella 'core'
set(COMMON_SOURCES
        core/main.c
        core/mpi_grid.c
        core/utils.c
        core/args.c
        core/validation.c
        cuda/kernel_cuda.cu
        omp/kernel_naive.c
)

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES native)
    add_executable(SCPA ${COMMON_SOURCES} cuda/kernel_cuda.cu)
    target_compile_definitions(SCPA PRIVATE USE_CUDA_BACKEND)
else()
    find_package(OpenMP REQUIRED)
    add_executable(SCPA ${COMMON_SOURCES} omp/kernel.c)
    target_link_libraries(SCPA PUBLIC OpenMP::OpenMP_C)
endif()

target_include_directories(SCPA PUBLIC include)
find_package(MPI REQUIRED)
target_link_libraries(SCPA PUBLIC MPI::MPI_C m)