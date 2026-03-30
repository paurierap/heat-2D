# Make library:
add_library(spatial
  FiniteDifference2D.cpp
)

target_include_directories(spatial
  PUBLIC
  ${PROJECT_SOURCE_DIR}/heat2D/Mesh2D
  ${PROJECT_SOURCE_DIR}/heat2D/BoundaryCondition
)

target_link_libraries(spatial
  PUBLIC
  mesh
  Eigen3::Eigen
)

# Build SpatialDiscretization test
add_executable(SpatialDiscretization2D_test
  SpatialDiscretization2D.test.cpp
)

target_link_libraries(SpatialDiscretization2D_test
  PUBLIC
  spatial
  GTest::gtest_main
)

gtest_discover_tests(SpatialDiscretization2D_test)