cmake_minimum_required(VERSION 3.14)
project(heat2d)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Additional compiler flags
add_compile_options(-march=native)

# Google Tests:
include(FetchContent)
FetchContent_Declare(
  googletest
  #URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Try to find Eigen
find_package(Eigen3 REQUIRED)

if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen not found system-wide; fetching with FetchContent")

  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
  )
  FetchContent_MakeAvailable(eigen)
endif()

# Try to find OpenMP
find_package(OpenMP)

# Try to find Gmsh
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")
find_package(Gmsh QUIET)
option(HEAT2D_ENABLE_GMSH "Enable Gmsh support for unstructured meshes" ${Gmsh_FOUND})

# Add subdirectories
add_subdirectory(heat2D)

option(HEAT2D_BUILD_EXAMPLES "Build examples" OFF)
option(HEAT2D_BUILD_BENCHMARKS "Build benchmarks" OFF)

if(HEAT2D_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(HEAT2D_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()