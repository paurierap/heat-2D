# # Make library:
# add_library(integrator
#   TimeIntegrator.cpp
# )

# target_include_directories(integrator
#   PUBLIC
#   ${PROJECT_SOURCE_DIR}/heat2D/SpatialDiscretization2D
# )

# target_link_libraries(integrator
#   PUBLIC
#   spatial
# )

# # Build Time integrator test
# add_executable(TimeIntegrator_test
#   TimeIntegrator.test.cpp
# )

# target_link_libraries(TimeIntegrator_test
#   integrator
# )

# gtest_discover_tests(TimeIntegrator_test)