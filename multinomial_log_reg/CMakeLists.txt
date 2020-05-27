cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

PROJECT(logistic_regression CXX)

add_subdirectory(applications)
add_subdirectory(coordinator)
add_subdirectory(utils)

add_library(logistic_regression
  SHARED
  multinomial_log_reg.hpp
  multinomial_log_reg.cpp)
target_link_libraries(logistic_regression PUBLIC blas utils)
