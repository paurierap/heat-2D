# Make library:
add_library(mesh
  StructuredMesh2D.cpp
)

# Build Mesh2D test
add_executable(Mesh2D_test
  Mesh2D.test.cpp
)

target_include_directories(mesh
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(Mesh2D_test
  mesh
  GTest::gtest_main
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_shape.msh
    ${CMAKE_CURRENT_BINARY_DIR}/test_shape.msh
    COPYONLY
)

gtest_discover_tests(Mesh2D_test DISCOVERY_TIMEOUT 30)

# Verify gmsh is found
if(HEAT2D_ENABLE_GMSH)
    if(NOT Gmsh_FOUND)
        message(FATAL_ERROR "HEAT2D_ENABLE_GMSH=ON but Gmsh was not found.")
    endif()
    message(STATUS "Gmsh found -- UnstructuredMesh2D enabled.")
    
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
else()
    message(STATUS "Gmsh not found -- UnstructuredMesh2D disabled.")
endif()