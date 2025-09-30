# Make library:
add_library(boundary_condition
  NeumannBoundaryCondition.cpp 
)

target_include_directories(boundary_condition
  PRIVATE
  ${PROJECT_SOURCE_DIR}/heat2D/Mesh2D
)

target_link_libraries(boundary_condition
  PRIVATE
  mesh
  Eigen3::Eigen
)

# Build code
add_executable(BoundaryCondition_test
  BoundaryCondition.test.cpp
)

target_include_directories(BoundaryCondition_test
  PRIVATE
  ${PROJECT_SOURCE_DIR}/heat2D/Mesh2D
)

target_link_libraries(BoundaryCondition_test
  boundary_condition
  mesh
  Eigen3::Eigen
  GTest::gtest_main
)

gtest_discover_tests(BoundaryCondition_test)