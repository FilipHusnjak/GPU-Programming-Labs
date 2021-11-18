cmake_minimum_required(VERSION 3.20)
project(lab_1 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(exercise_1 exercise_1.cu)
set_target_properties(exercise_1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(exercise_2 exercise_2.cu)
set_target_properties(exercise_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(exercise_3 exercise_3.cu)
set_target_properties(exercise_3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(bonus_exercise bonus_exercise.cu)
set_target_properties(bonus_exercise PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
