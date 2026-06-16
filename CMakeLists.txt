cmake_minimum_required(VERSION 3.10)
#项目名称
project(my_cnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 开启最高级别的优化选项，这对于跑神经网络非常重要
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

#头文件.hpp目录
include_directories(include)

#.cpp 源文件
file(GLOB SOURCES "src/*.cpp")

add_executable(my_cnn ${SOURCES})