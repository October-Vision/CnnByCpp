cmake_minimum_required(VERSION 3.18)

project(my_cnn LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDAToolkit REQUIRED)


find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(include ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "src/*.cpp" "src/*.cu")

add_executable(my_cnn ${SOURCES})

# 极其清爽的链接，不需要再手写 Qt 依赖
target_link_libraries(my_cnn PRIVATE 
    CUDA::cublas 
    opencv_core
    opencv_imgproc
    opencv_highgui
)

target_compile_options(my_cnn PRIVATE -O3 -g -Wall)