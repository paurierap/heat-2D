cmake_minimum_required(VERSION 3.14)
project(heat2d)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#=============================================================================
# Options
#=============================================================================

# Detect whether this is the top-level project or not (for building tests)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(HEAT2D_IS_TOP_LEVEL ON)
else()
  set(HEAT2D_IS_TOP_LEVEL OFF)
endif()

option(HEAT2D_ENABLE_NATIVE_ARCH "Build with -march=native" OFF)
option(HEAT2D_ENABLE_OPENMP "Build with OpenMP support" OFF)
option(HEAT2D_BUILD_TESTS "Build tests" ${HEAT2D_IS_TOP_LEVEL})
option(HEAT2D_BUILD_EXAMPLES "Build examples" OFF)
option(HEAT2D_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(HEAT2D_FETCH_DEPENDENCIES "Download missing dependencies" ON)

if(HEAT2D_ENABLE_NATIVE_ARCH)
  add_compile_options(-march=native)
endif()

#==============================================================================
# External dependencies
#==============================================================================

# Google Test
if(HEAT2D_BUILD_TESTS)
  find_package(GTest QUIET)

  if(NOT TARGET GTest::gtest_main)
    if(NOT HEAT2D_FETCH_DEPENDENCIES)
      message(FATAL_ERROR "Install GoogleTest or enable HEAT2D_FETCH_DEPENDENCIES (or disable HEAT2D_BUILD_TESTS).")
    endif()
    message(STATUS "GTest not found system-wide; fetching with FetchContent")

    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG v1.17.0
    )
    FetchContent_MakeAvailable(googletest)
  endif()

  enable_testing()
  include(GoogleTest)
endif()

# Eigen
find_package(Eigen3 QUIET)

if(NOT TARGET Eigen3::Eigen)
  if(NOT HEAT2D_FETCH_DEPENDENCIES)
    message(FATAL_ERROR "Install Eigen3 or enable HEAT2D_FETCH_DEPENDENCIES.")
  endif()
  message(STATUS "Eigen not found system-wide; fetching with FetchContent")

  include(FetchContent)
  FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.0
  )
  FetchContent_MakeAvailable(eigen)
endif()

# VTU compression: prefer system zlib, otherwise build a pinned release.
find_package(ZLIB QUIET)
if(NOT TARGET ZLIB::ZLIB)
  if(NOT HEAT2D_FETCH_DEPENDENCIES)
    message(FATAL_ERROR "Install zlib development headers/libraries (e.g. zlib1g-dev on Debian/Ubuntu), set ZLIB_ROOT, or enable HEAT2D_FETCH_DEPENDENCIES.")
  endif()
  include(FetchContent)
  set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "Build zlib examples" FORCE)
  FetchContent_Declare(
    heat2d_zlib
    URL https://zlib.net/fossils/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
  )
  FetchContent_MakeAvailable(heat2d_zlib)
  # Upstream 1.3.1 exposes zlibstatic, not ZLIB::ZLIB.
  set_target_properties(zlib PROPERTIES EXCLUDE_FROM_ALL TRUE)
  set_target_properties(zlibstatic PROPERTIES POSITION_INDEPENDENT_CODE ON)
  add_library(ZLIB::ZLIB ALIAS zlibstatic)
endif()

if(HEAT2D_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# Gmsh
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")
find_package(Gmsh QUIET)
option(HEAT2D_ENABLE_GMSH "Enable Gmsh support for unstructured meshes" ${Gmsh_FOUND})

#=============================================================================
# Add subdirectories
#=============================================================================

add_subdirectory(heat2D)

if(HEAT2D_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(HEAT2D_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
