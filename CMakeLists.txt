cmake_minimum_required(VERSION 3.14)
project(heat2d)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# Add subdirectories
add_subdirectory(tests)
add_subdirectory(heat2D)