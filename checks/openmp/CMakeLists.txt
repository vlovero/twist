cmake_minimum_required(VERSION 3.24)

project(check_openmp
    VERSION 0.1.0
    DESCRIPTION "Check existence of OpenMP"
    LANGUAGES CXX
)

find_package(OpenMP REQUIRED)