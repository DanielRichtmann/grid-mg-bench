# FindGrid
#
# Copyright (c) 2020, Daniel Richtmann
#
# Before this module is called, the cmake variable "GRID_DIR" must hold the
# path to a directory containing a grid-config binary. This module then calls this
# binary to define the following variables:
#
#  Grid_FOUND - Whether Grid was correctly detected
#  Grid_PREFIX - The installation prefix for Grid
#  Grid_CXX - The compiler used to build Grid
#  Grid_CXXLD - The linker used to link Grid
#  Grid_CXXFLAGS - The cxx flags used to build Grid
#  Grid_LDFLAGS - The linker flags used to build Grid
#  Grid_LIBS - The libraries Grid uses
#  Grid_VERSION - The version of Grid
#  Grid_GIT - The branch and commit hash
#  Grid_SUMMARY - A summary of the above

macro(callconfigbinary binary flag var)
    execute_process(COMMAND "${binary}" "${flag}"
        RESULT_VARIABLE tmpresult
        OUTPUT_VARIABLE ${var}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)
    if(${tmpresult} EQUAL 0)
        message(STATUS "${binary} ${flag} = ${${var}}")
    else()
        message(FATAL_ERROR "${binary} ${flag} failed")
    endif()
endmacro()

if(GRID_DIR STREQUAL "")
    message(FATAL_ERROR "FindGrid.cmake: Variable \"GRID_DIR\" not set. Please use cmake -DGRID_DIR=/path/to/grid")
endif()

set(grid_config_binary "${GRID_DIR}/grid-config")

if(EXISTS "${grid_config_binary}")
    set(Grid_FOUND TRUE)
    callconfigbinary(${grid_config_binary} --prefix Grid_PREFIX)
    callconfigbinary(${grid_config_binary} --cxx Grid_CXX)
    callconfigbinary(${grid_config_binary} --cxxld Grid_CXXLD)
    callconfigbinary(${grid_config_binary} --cxxflags Grid_CXXFLAGS)
    callconfigbinary(${grid_config_binary} --ldflags Grid_LDFLAGS)
    callconfigbinary(${grid_config_binary} --libs Grid_LIBS)
    callconfigbinary(${grid_config_binary} --version Grid_VERSION)
    callconfigbinary(${grid_config_binary} --git Grid_GIT)
    callconfigbinary(${grid_config_binary} --summary Grid_SUMMARY)
else()
    set(Grid_FOUND FALSE)
    message(FATAL_ERROR "FindGrid.cmake: grid config binary \"${grid_config_binary}\" not found")
endif()
