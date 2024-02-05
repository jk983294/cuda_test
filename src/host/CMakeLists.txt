cmake_minimum_required(VERSION 3.17)

project(my_cuda_mix)
enable_language(CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -O3 -s -Wall -Werror -Wno-format-truncation")
#set(CMAKE_CXX_FLAGS "-std=c++17 -g -Wall -Werror")
#SET(CMAKE_CXX_FLAGS "-std=c++17 -g -fsanitize=address -Wall -Werror")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

link_directories(/usr/local/cuda/lib64/)

file( GLOB APP_SOURCES "*.cpp")
add_executable(my_host ${APP_SOURCES})
# target_link_libraries(my_host mycudalib_static cudart)
target_link_libraries(my_host mycudalib_static cudart_static pthread dl rt)
