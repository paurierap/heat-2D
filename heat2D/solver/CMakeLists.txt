# Make library:
add_library(solver
  FiniteDifference2D.cpp
)

target_include_directories(solver
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(solver
  PUBLIC
  mesh
  bc
  Eigen3::Eigen
  $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)

if(HEAT2D_BUILD_TESTS)
  # Build SpatialDiscretization test
  add_executable(SpatialDiscretization2D_test
    SpatialDiscretization2D.test.cpp
  )

  target_link_libraries(SpatialDiscretization2D_test
    PUBLIC
    solver
    GTest::gtest_main
  )

  # Increase the timeout for test discovery to avoid issues on slow systems.
  gtest_discover_tests(SpatialDiscretization2D_test DISCOVERY_TIMEOUT 30)
endif()
