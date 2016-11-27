# cmake_minimum_required (VERSION 2.8)
# project (dfftlib)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
set (dfftlib_VERSION_MAJOR 0)
set (dfftlib_VERSION_MINOR 1)

# already setup by hoomd
# if (NOT CMAKE_BUILD_TYPE)
#     message(STATUS "No build type selected, default to Release")
#     set(CMAKE_BUILD_TYPE "Release")
# endif()

# MPI already setup by hoomd
# find_package(MPI REQUIRED)

include(FindLocalFFT.cmake)

# OpenMP not supported by hoomd
# configure OpenMP support
# if (ENABLE_OPENMP)
#     find_package(OpenMP)
#     if(OPENMP_FOUND)
#     message("OpenMP FOUND")
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#     add_definitions(-DENABLE_OPENMP)
#     endif()
# endif()

# already setup by hoomd
#find_package(CUDA)
#if(CUDA_FOUND)
#    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-arch=sm_20")
#    option(ENABLE_CUDA "CUDA support" ON)
#else()
#    option(ENABLE_CUDA "CUDA support" OFF)
#endif()

# include(CUDA_MPI.cmake)

configure_file (
  "${CMAKE_CURRENT_SOURCE_DIR}/src/dfft_lib_config.h.in"
  "${HOOMD_BINARY_DIR}/hoomd/include/dfft_lib_config.h"
   )

install(FILES "${HOOMD_BINARY_DIR}/hoomd/include/dfft_lib_config.h" DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd/extern/dfftlib/src)

# Using manual source configuration below
# library source
# add_subdirectory(src)

# unit tests
#add_subdirectory(test)

if(ENABLE_CUDA)
    # CUFFT is default for CUDA
    set(CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/cufft_single_interface.c)

    set(CUDA_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/dfft_cuda.c ${CUDA_SOURCES})
    # cuda_compile(DFFT_CUDA_O ${CMAKE_CURRENT_SOURCE_DIR}/src/dfft_cuda.cu OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
    # set(CUDA_OBJECTS ${DFFT_CUDA_O})
    # set(CUDA_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_cufft_LIBRARY})
    set(DFFT_CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/dfft_cuda.cu PARENT_SCOPE)
endif()

if(ENABLE_HOST)
    if(LOCAL_FFT_LIB STREQUAL "LOCAL_LIB_MKL")
        set(HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/mkl_single_interface.c)
    elseif(LOCAL_FFT_LIB STREQUAL "LOCAL_LIB_ACML")
        set(HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/acml_single_interface.c)
    elseif(LOCAL_FFT_LIB STREQUAL "LOCAL_LIB_BARE")
        set(HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/bare_fft_interface.c ${CMAKE_CURRENT_SOURCE_DIR}/src/bare_fft.c)
    endif()

    set(HOST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/dfft_host.c ${HOST_SOURCES})
endif()

set(DFFT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/dfft_common.c ${HOST_SOURCES} ${CUDA_SOURCES} PARENT_SCOPE)
