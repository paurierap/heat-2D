cmake_minimum_required(VERSION 3.14)

include(FetchContent)
FetchContent_Declare(
  googletest
  #URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Build code
add_executable(
  Mesh2D_test
  Mesh2D.test.cpp
)

target_link_libraries(
  Mesh2D_test
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(Mesh2D_test)