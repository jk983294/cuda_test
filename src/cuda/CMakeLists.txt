cmake_minimum_required(VERSION 3.17)

project(cuda_lib)
#set(CMAKE_CUDA_ARCHITECTURES 65)
#set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
else()
    message("No OpenMP detected, fstlib builds without OpenMP but needs it for optimal performance!")
endif()

set(CUDA_NVCC_FLAGS "-O3 -DNDEBUG")

file( GLOB CUDA_LIB_SOURCES "*.cu")
add_library(mycudalib SHARED ${CUDA_LIB_SOURCES})
add_library(mycudalib_static STATIC ${CUDA_LIB_SOURCES})
set_target_properties(mycudalib_static PROPERTIES 
                        CUDA_SEPARABLE_COMPILATION ON
                        CUDA_RESOLVE_DEVICE_SYMBOLS ON
                        POSITION_INDEPENDENT_CODE ON)

add_subdirectory(play)