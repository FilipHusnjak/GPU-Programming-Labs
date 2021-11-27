cmake_minimum_required(VERSION 3.20)
project(Assignment_3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Assignment_3 ex_1/hw3_ex1.cu)

set_target_properties(Assignment_3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
