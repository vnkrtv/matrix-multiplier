cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(matrix_mapreduce LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -lstdc++)

find_package(CUDA REQUIRED)

find_package(Qt5 COMPONENTS Concurrent REQUIRED)

set(CMAKE_PREFIX_PATH /opt/pytorch/libtorch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(matrix_mapreduce src/main.cpp src/mapreduce.hpp src/configurator.hpp src/matrix.hpp src/args.hpp)

target_link_libraries(matrix_mapreduce Qt5::Concurrent)
target_link_libraries(matrix_mapreduce "${TORCH_LIBRARIES}")
target_link_libraries(matrix_mapreduce ${CUDA_LIBRARIES})
