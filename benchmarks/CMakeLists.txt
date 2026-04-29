# Build benchmark
add_executable(benchmark
  benchmark.cpp
)

target_link_libraries(benchmark 
  HeatPDE
)