#=============================================================================
# quadrature - 1D and 2D quadrature rules
#=============================================================================

# Dunavant rules for triangles
add_library(triangle_quadrature STATIC
  ${CMAKE_SOURCE_DIR}/external/dunavant/triangle_dunavant_rule.cpp
)

# Public quadrature interface (header-only)
add_library(quadrature INTERFACE)

target_include_directories(quadrature
  INTERFACE
  ${CMAKE_SOURCE_DIR}/external/dunavant
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(quadrature
  INTERFACE
  triangle_quadrature
  Eigen3::Eigen
)

if(HEAT2D_BUILD_TESTS)
  add_executable(Quadrature_test
    Quadrature.test.cpp
  )

  target_link_libraries(Quadrature_test
    quadrature
    GTest::gtest_main
  )

  gtest_discover_tests(Quadrature_test)
endif()
