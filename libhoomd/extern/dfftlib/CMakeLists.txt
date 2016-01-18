cmake_minimum_required (VERSION 2.8)
#project (dfftlib)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
set (dfftlib_VERSION_MAJOR 0)
set (dfftlib_VERSION_MINOR 1)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(MPI REQUIRED)
include(FindLocalFFT.cmake)

# configure OpenMP support
if (ENABLE_OPENMP)
    find_package(OpenMP)
    if(OPENMP_FOUND)
    message("OpenMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    add_definitions(-DENABLE_OPENMP)
    endif()
endif()

#find_package(CUDA)
#if(CUDA_FOUND)
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_20")
#    option(ENABLE_CUDA "CUDA support" ON)
#else()
#    option(ENABLE_CUDA "CUDA support" OFF)
#endif()

include(CUDA_MPI.cmake)

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/src/dfft_lib_config.h.in"
  "${PROJECT_BINARY_DIR}/dfft_lib_config.h"
   )

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}")

# library source
add_subdirectory(src)

# unit tests
#add_subdirectory(test)

install(FILES "${PROJECT_BINARY_DIR}/dfft_lib_config.h" DESTINATION "include")
