#=============================================================================
# mesh - structured / unstructured meshes
#=============================================================================

add_library(mesh
  StructuredMesh2D.cpp
)

target_include_directories(mesh
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optional Gmsh support; enables UnstructuredMesh2D
if(HEAT2D_ENABLE_GMSH)
  if(NOT Gmsh_FOUND)
    message(FATAL_ERROR "HEAT2D_ENABLE_GMSH=ON but Gmsh was not found.")
  endif()

  target_sources(mesh
    PRIVATE
    UnstructuredMesh2D.cpp
  )
  target_link_libraries(mesh
    PRIVATE
    Gmsh::Gmsh
  )
  target_compile_definitions(mesh
    PUBLIC
    HEAT2D_HAS_GMSH
  )
endif()

if(HEAT2D_BUILD_TESTS)
  add_executable(Mesh2D_test
    Mesh2D.test.cpp
  )

  target_link_libraries(Mesh2D_test
    mesh
    GTest::gtest_main
  )

  # Required for tests that load test_shape.msh
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_shape.msh
    ${CMAKE_CURRENT_BINARY_DIR}/test_shape.msh
    COPYONLY
  )

  # Increase discovery timeout to avoid issues on slow systems
  gtest_discover_tests(Mesh2D_test DISCOVERY_TIMEOUT 30)
endif()
