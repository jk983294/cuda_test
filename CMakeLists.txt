cmake_minimum_required(VERSION 3.17)

set(CMAKE_C_COMPILER /usr/bin/gcc-8)
set(CMAKE_CXX_COMPILER /usr/bin/g++-8)

project(cuda_test)

SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
message(STATUS "OpenCL include directory: " ${OpenCL_INCLUDE_DIRS})

set(EIGEN_DIR "/opt/3rd/Eigen")
include_directories(SYSTEM ${EIGEN_DIR})
add_definitions(-DEIGEN_MPL2_ONLY)
add_definitions(-DEIGEN_DONT_PARALLELIZE)

include_directories(include)
include_directories(third/compute/include)

add_subdirectory(src/host)
add_subdirectory(src/cuda)
add_subdirectory(src/third/compute_examples)
