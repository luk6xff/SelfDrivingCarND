project(LaserMeasurements)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

set(sources main.cpp tracking.cpp kalman_filter.cpp tracking.h kalman_filter.h measurement_package.h)

add_executable(Laser-Measurements ${sources})

