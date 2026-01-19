# Build code
add_executable(BoundaryCondition_test
  BoundaryCondition.test.cpp
)

target_link_libraries(BoundaryCondition_test
  GTest::gtest_main
)

gtest_discover_tests(BoundaryCondition_test)