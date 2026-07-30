# Add header-only library
add_library(ode INTERFACE)

target_include_directories(ode
  INTERFACE 
  ${CMAKE_CURRENT_SOURCE_DIR}
)