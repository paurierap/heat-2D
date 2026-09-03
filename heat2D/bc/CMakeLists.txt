#=============================================================================
# bc - boundary conditions (header-only)
#=============================================================================

add_library(bc INTERFACE)

target_include_directories(bc
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

if(HEAT2D_BUILD_TESTS)
  add_executable(BoundaryCondition_test
    BoundaryCondition.test.cpp
  )

  target_link_libraries(BoundaryCondition_test
    GTest::gtest_main
  )

  gtest_discover_tests(BoundaryCondition_test)
endif()
