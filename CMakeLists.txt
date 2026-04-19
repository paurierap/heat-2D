cmake_minimum_required(VERSION 3.14)
project(heat2d)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# Google Tests:
include(FetchContent)
FetchContent_Declare(
  googletest
  #URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Try to find Eigen in the system
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

if(NOT TARGET Eigen3::Eigen)
  message(STATUS "Eigen not found system-wide; fetching with FetchContent")

  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(eigen)
endif()

# Add subdirectories
add_subdirectory(heat2D)

option(BUILD_EXAMPLES "Build examples" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()