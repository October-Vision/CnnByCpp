cmake_minimum_required(VERSION 3.18)

project(my_cnn LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)

#头文件
include_directories(include)

#.cpp和.cu源文件
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

#生成可执行文件
add_executable(my_cnn ${SOURCES})

target_link_libraries(my_cnn PRIVATE CUDA::cublas)

target_compile_options(my_cnn PRIVATE -O3 -g -Wall)