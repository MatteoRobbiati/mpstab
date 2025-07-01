cmake_minimum_required(VERSION 3.14)
project(c_tensor_network)

include(FetchContent)
# Fetch pybind11
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.10.4
)
FetchContent_MakeAvailable(pybind11)

# Fetch Eigen (header-only)
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(eigen)

# Build extension
pybind11_add_module(c_tensor_network
    c_tensor_network.cpp
    bindings.cpp
)
# Include Eigen headers
target_include_directories(c_tensor_network PRIVATE ${eigen_SOURCE_DIR})
set_target_properties(c_tensor_network PROPERTIES
  PREFIX ""
  OUTPUT_NAME "c_tensor_network"
)
