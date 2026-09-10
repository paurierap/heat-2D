#=============================================================================
# solver - spatial discretization (finite difference / finite element)
#=============================================================================

add_library(solver
  SpatialDiscretization2D.cpp
  fdm/FiniteDifference2D.cpp
  fem/FiniteElement2D.cpp
)

target_include_directories(solver
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(solver
  PUBLIC
  mesh
  bc
  quadrature
  Eigen3::Eigen
  $<$<BOOL:${HEAT2D_ENABLE_OPENMP}>:OpenMP::OpenMP_CXX>
)

if(HEAT2D_BUILD_TESTS)
  add_executable(SpatialDiscretization2D_test
    SpatialDiscretization2D.test.cpp
  )

  target_link_libraries(SpatialDiscretization2D_test
    PUBLIC
    solver
    GTest::gtest_main
  )

  # Increase discovery timeout to avoid issues on slow systems
  gtest_discover_tests(SpatialDiscretization2D_test DISCOVERY_TIMEOUT 30)
endif()
