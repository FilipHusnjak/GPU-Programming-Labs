cmake_minimum_required(VERSION 3.20)
project(Assignment_2 CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(exercise_1 ex_1/exercise_1.cu)
set_target_properties(exercise_1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(exercise_2 ex_2/exercise_2.cu)
set_target_properties(exercise_2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(exercise_3 ex_3/exercise_3.cu)
set_target_properties(exercise_3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(bonus_exercise ex_bonus/exercise_bonus.cu)
set_target_properties(bonus_exercise PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
