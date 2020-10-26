# Maintainer: joaander

include_directories(${PYTHON_INCLUDE_DIR})

################################
## Define common libraries used by every target in HOOMD

option(ENABLE_TBB "Enable support for Threading Building Blocks (TBB)" off)

if(ENABLE_TBB)
    find_package(TBB 4.3)
    include_directories(${TBB_INCLUDE_DIR})

    # Detect clang and fix incompatibility with TBB
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

set(HOOMD_COMMON_LIBS ${ADDITIONAL_LIBS})

if (ENABLE_TBB)
    list(APPEND HOOMD_COMMON_LIBS ${TBB_LIBRARY})
endif()

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
