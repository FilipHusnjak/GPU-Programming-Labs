cmake_minimum_required(VERSION 3.20)
project(Assignment_3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif ()

add_executable(exercise1 ex_1/hw3_ex1.cu)

set_target_properties(exercise1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
