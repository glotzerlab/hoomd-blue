# Maintainer: joaander

include_directories(${HOOMD_PYTHON_INCLUDE_DIR})

################################
## Define common libraries used by every target in HOOMD

## An update to to CentOS5's python broke linking of the hoomd exe. According
## to an ancient post online, adding -lutil fixed this in python 2.2
set(ADDITIONAL_LIBS "")
if (UNIX AND NOT APPLE)
    find_library(UTIL_LIB util /usr/lib)
    find_library(DL_LIB dl /usr/lib)
    set(ADDITIONAL_LIBS ${UTIL_LIB} ${DL_LIB})
    if (DL_LIB AND UTIL_LIB)
    mark_as_advanced(UTIL_LIB DL_LIB)
    endif (DL_LIB AND UTIL_LIB)
endif (UNIX AND NOT APPLE)

option(ENABLE_TBB "Enable support for Threading Building Blocks (TBB)" off)

if(ENABLE_TBB)
    # find TBB lib and includes
    find_library(TBB_LIBRARY tbb
                 PATHS ENV TBB_LINK)
    find_path(TBB_INCLUDE_DIR tbb/tbb.h
              PATHS ENV TBB_INC)
    include_directories(${TBB_INCLUDE_DIR})
    if (TBB_LIBRARY)
        mark_as_advanced(TBB_LIBRARY)
    endif()
    if (TBB_INCLUDE_DIR)
        mark_as_advanced(TBB_INCLUDE_DIR)
    endif()
    if (TBB_INCLUDE_DIR AND TBB_LIBRARY)
        add_definitions(-DENABLE_TBB)
    endif()

    # Detect clang and fix incompatiblity with TBB
    # https://github.com/wjakob/tbb/blob/master/CMakeLists.txt
    if (NOT TBB_USE_GLIBCXX_VERSION AND UNIX AND NOT APPLE)
      if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # using Clang
        string(REPLACE "." "0" TBB_USE_GLIBCXX_VERSION ${CMAKE_CXX_COMPILER_VERSION})
      endif()
    endif()
endif()

if (TBB_USE_GLIBCXX_VERSION)
   add_definitions(-DTBB_USE_GLIBCXX_VERSION=${TBB_USE_GLIBCXX_VERSION})
endif()

set(HOOMD_COMMON_LIBS
        ${ADDITIONAL_LIBS}
        ${TBB_LIBRARY}
        )

if (APPLE)
    list(APPEND HOOMD_COMMON_LIBS "-undefined dynamic_lookup")
endif()

if (ENABLE_CUDA)
    list(APPEND HOOMD_COMMON_LIBS ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY} ${CUDA_curand_LIBRARY})

    if (ENABLE_NVTOOLS)
        list(APPEND HOOMD_COMMON_LIBS ${CUDA_nvToolsExt_LIBRARY})
    endif()
endif (ENABLE_CUDA)

if (ENABLE_MPI)
    list(APPEND HOOMD_COMMON_LIBS ${MPI_CXX_LIBRARIES})
endif (ENABLE_MPI)
