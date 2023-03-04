cmake_minimum_required(VERSION 3.16)
project(lab2_OpenMP C)

set(CMAKE_C_STANDARD 99)
SET(CMAKE_C_FLAGS -fopenmp)

add_executable(lab2_OpenMP  task2.c)
