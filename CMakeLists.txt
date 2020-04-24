cmake_minimum_required(VERSION 3.6.3 FATAL_ERROR)

project(grid-mg-bench LANGUAGES CXX)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(GRID_DIR "" CACHE PATH "path to a directory holding a grid-config binary")
find_package(Grid REQUIRED)
if(Grid_FOUND)
    set(CMAKE_CXX_COMPILER ${Grid_CXX})
    set(CMAKE_CXX_FLAGS        "-I${Grid_PREFIX}/include ${Grid_CXXFLAGS} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "-L${Grid_PREFIX}/lib ${Grid_LDFLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
endif()

include_directories(include)
add_subdirectory(src)
