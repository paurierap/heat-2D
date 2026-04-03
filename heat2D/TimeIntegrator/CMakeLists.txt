# Add header-only library
add_library(temporal INTERFACE)

target_include_directories(temporal
  INTERFACE 
  ${CMAKE_CURRENT_SOURCE_DIR}
)