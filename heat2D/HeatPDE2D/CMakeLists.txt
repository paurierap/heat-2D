# Add header-only library
add_library(HeatPDE INTERFACE)

target_include_directories(HeatPDE 
  INTERFACE 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/heat2D/TimeIntegrator
  ${PROJECT_SOURCE_DIR}/heat2D/SolutionWriter
)

target_link_libraries(HeatPDE 
  INTERFACE 
  spatial 
  temporal
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