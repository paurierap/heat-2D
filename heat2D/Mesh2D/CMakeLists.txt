# Make library:
add_library(mesh
  Mesh2D.cpp 
  StructuredMesh2D.cpp
)

target_include_directories(mesh
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Build Mesh2D test
add_executable(Mesh2D_test
  Mesh2D.test.cpp
)

target_link_libraries(Mesh2D_test
  mesh
  GTest::gtest_main
)

gtest_discover_tests(Mesh2D_test)