# Make library:
add_library(HeatPDE
  HeatPDE2D.cpp
)

target_include_directories(HeatPDE
  PUBLIC
  ${PROJECT_SOURCE_DIR}/heat2D/SpatialDiscretization2D
  ${PROJECT_SOURCE_DIR}/heat2D/TimeIntegrator
)

target_link_libraries(HeatPDE
  PUBLIC
  spatial
)

# Build HeatPDE test
add_executable(HeatPDE2D_test
    HeatPDE2D.test.cpp
)

target_link_libraries(HeatPDE2D_test
  HeatPDE
  GTest::gtest_main
)

gtest_discover_tests(HeatPDE2D_test)