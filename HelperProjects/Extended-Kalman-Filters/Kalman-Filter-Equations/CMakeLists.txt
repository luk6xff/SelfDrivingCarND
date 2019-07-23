project(KalmanFilterEquations)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources kalman-filter-equations.cpp)

add_executable(KalmanFilterEquations ${sources})

