cmake_minimum_required(VERSION 3.0)

add_executable(async_mnist_log_reg async_mnist_log_reg.cpp)
target_include_directories(async_mnist_log_reg
  PUBLIC
  ${logistic_regression_SOURCE_DIR})
target_link_libraries(async_mnist_log_reg coordinator utils)

add_executable(parameter_server parameter_server.cpp)
target_include_directories(parameter_server
  PUBLIC
  ${logistic_regression_SOURCE_DIR})
target_link_libraries(parameter_server coordinator logistic_regression utils)
