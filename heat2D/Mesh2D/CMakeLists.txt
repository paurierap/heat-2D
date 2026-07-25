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

gtest_discover_tests(Mesh2D_test)

# Verify gmsh is found
if(ENABLE_GMSH)
    if(NOT gmsh_FOUND)
        message(FATAL_ERROR "ENABLE_GMSH=ON but Gmsh was not found.")
    endif()
    message(STATUS "Gmsh found -- UnstructuredMesh2D enabled.")
    
    # target_sources(mesh 
    #   PRIVATE 
    #   UnstructuredMesh2D.cpp
    # )
    # target_link_libraries(mesh 
    #   PRIVATE 
    #   gmsh
    # )
    # target_compile_definitions(mesh 
    #   PRIVATE
    #   HEAT2D_HAS_GMSH
    # )

else()
    message(STATUS "Gmsh not found -- UnstructuredMesh2D disabled.")
endif()

