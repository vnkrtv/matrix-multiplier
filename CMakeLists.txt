cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(matrix_computer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS -lstdc++)

find_package(CUDA REQUIRED)
find_package(Qt5 COMPONENTS Concurrent REQUIRED)
find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} src/main.cpp src/mapreduce.hpp src/configurator.hpp src/matrix.hpp src/args.hpp)

target_link_libraries(${PROJECT_NAME} Qt5::Concurrent)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
