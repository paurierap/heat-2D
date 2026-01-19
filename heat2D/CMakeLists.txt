cmake_minimum_required(VERSION 3.14)

# Add subdirectories
add_subdirectory(BoundaryCondition)
add_subdirectory(Mesh2D)
add_subdirectory(SpatialDiscretization2D)

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

# Build code
