cmake_minimum_required(VERSION 3.0)

add_library(utils SHARED
  utils.hpp utils.cpp
  ml_stat.hpp ml_stat.cpp
  cnpy.hpp cnpy.cpp
  numpy_reader.hpp numpy_reader.cpp)
target_include_directories(utils PUBLIC ${logistic_regression_SOURCE_DIR})
target_link_libraries(utils PUBLIC blas)
