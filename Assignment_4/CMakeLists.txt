cmake_minimum_required(VERSION 3.20)
project(Assignment_4)

include_directories("$ENV{CUDA_PATH}/include")

add_executable(exercise_1 ex_1/exercise_1.cpp)
add_executable(exercise_2 ex_2/exercise_2.cpp)
add_executable(exercise_3 ex_3/exercise_3.c)

target_link_libraries(exercise_1 "$ENV{CUDA_PATH}/lib/x64/OpenCL.lib")
target_link_libraries(exercise_2 "$ENV{CUDA_PATH}/lib/x64/OpenCL.lib")
target_link_libraries(exercise_3 "$ENV{CUDA_PATH}/lib/x64/OpenCL.lib")
