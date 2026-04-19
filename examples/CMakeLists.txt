# Build example executables (one per .cpp file)
file(GLOB EXAMPLE_SOURCES "*.cpp")

foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
    get_filename_component(EXEC_NAME ${SOURCE_FILE} NAME_WE)

    add_executable(${EXEC_NAME} 
      ${SOURCE_FILE}
    )
    
    target_link_libraries(${EXEC_NAME} 
      HeatPDE
    )
endforeach()