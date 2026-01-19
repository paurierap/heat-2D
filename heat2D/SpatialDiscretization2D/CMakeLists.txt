# Make library:
add_library(spatial
  FiniteDifference2D.cpp
)

target_include_directories(spatial
  PRIVATE
  ${PROJECT_SOURCE_DIR}/heat2D/Mesh2D
  ${PROJECT_SOURCE_DIR}/heat2D/BoundaryCondition
)

target_link_libraries(spatial
  PRIVATE
  mesh
  Eigen3::Eigen
)

# Build code
add_executable(SpatialDiscretization2D_test
  SpatialDiscretization2D.test.cpp
)

target_include_directories(SpatialDiscretization2D_test
  PRIVATE
  ${PROJECT_SOURCE_DIR}/heat2D/Mesh2D
  ${PROJECT_SOURCE_DIR}/heat2D/BoundaryCondition
)

target_link_libraries(SpatialDiscretization2D_test
  spatial
  mesh
  Eigen3::Eigen
  GTest::gtest_main
)

gtest_discover_tests(SpatialDiscretization2D_test)