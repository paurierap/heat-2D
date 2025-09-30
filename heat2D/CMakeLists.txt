cmake_minimum_required(VERSION 3.14)

# Add subdirectories
add_subdirectory(BoundaryCondition)
add_subdirectory(Mesh2D)
add_subdirectory(SpatialDiscretization2D)

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

# Make library:
add_library(heatPDE
  heatPDE.cpp
)

target_include_directories(heatPDE
  PRIVATE
  ${PROJECT_SOURCE_DIR}/heat2D/Mesh2D
  ${PROJECT_SOURCE_DIR}/heat2D/BoundaryCondition
  ${PROJECT_SOURCE_DIR}/heat2D/SpatialDiscretization2D
  ${PROJECT_SOURCE_DIR}/heat2D/TimeIntegrator
)

target_link_libraries(heatPDE
  PRIVATE
  mesh
  spatial
  Eigen3::Eigen
)

# Google Tests:
include(FetchContent)
FetchContent_Declare(
  googletest
  #URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

# Build code
