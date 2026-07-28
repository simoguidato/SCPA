cmake_minimum_required(VERSION 3.22)
project(SCPA LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_CUDA "Abilita l'accelerazione GPU con CUDA" OFF)
set(CUDA_KERNEL "4" CACHE STRING "Kernel CUDA da compilare (0, 1, 2, 3 oppure 4)")
set(SCPA_CUDA_ARCHITECTURES "80" CACHE STRING
    "Compute capability CUDA, per esempio 80, 86 o 89")

set(CORE_SOURCES
    core/main.c
    core/mpi_grid.c
    core/utils.c
    core/args.c
    core/validation.c
)

if(USE_CUDA)
    if(NOT CUDA_KERNEL MATCHES "^[0-4]$")
        message(FATAL_ERROR "CUDA_KERNEL deve essere un valore tra 0 e 4")
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "${SCPA_CUDA_ARCHITECTURES}")
    add_executable(SCPA ${CORE_SOURCES} cuda/kernel_cuda_${CUDA_KERNEL}.cu)
    target_compile_options(SCPA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_compile_definitions(SCPA PRIVATE USE_CUDA_BACKEND)
else()
    find_package(OpenMP REQUIRED)
    add_executable(SCPA ${CORE_SOURCES} omp/kernel.c omp/kernel_naive.c)
    target_link_libraries(SCPA PUBLIC OpenMP::OpenMP_C)
endif()

target_include_directories(SCPA PUBLIC include cuda)
find_package(MPI REQUIRED)
target_link_libraries(SCPA PUBLIC MPI::MPI_C m)
