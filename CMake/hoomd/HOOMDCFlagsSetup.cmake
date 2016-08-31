# Maintainer: joaander

#################################
## Setup default CXXFLAGS
if(NOT PASSED_FIRST_CONFIGURE)
    message(STATUS "Overriding CMake's default CFLAGS (this should appear only once)")

    ## Allow GCC_ARCH flag to set the -march field
    if(NOT GCC_ARCH AND "$ENV{GCC_ARCH}" STREQUAL "")
        set(GCC_ARCH "native")
        message(STATUS "GCC_ARCH env var not set, setting -march to ${GCC_ARCH}")
    else()
        set(GCC_ARCH $ENV{GCC_ARCH})
        message(STATUS "Found GCC_ARCH env var, setting -march to ${GCC_ARCH}")
    endif()


    # default build type is Release when compiling make files
    if(NOT CMAKE_BUILD_TYPE)
       if(${CMAKE_GENERATOR} STREQUAL "Xcode")

       else(${CMAKE_GENERATOR} STREQUAL "Xcode")
            set(CMAKE_BUILD_TYPE "Release" CACHE STRING  "Build type: options are None, Release, Debug, RelWithDebInfo" FORCE)
        endif(${CMAKE_GENERATOR} STREQUAL "Xcode")
    endif()

    if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

        set(_common_options "-march=${GCC_ARCH} -Wall -Wno-unknown-pragmas")
        set(_common_cxx_options "-march=${GCC_ARCH} -Wall -Wno-unknown-pragmas")

        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            set(_common_cxx_options "${_common_cxx_options} -Wno-c++14-extensions")
        endif()

        # default flags for g++
        set(CMAKE_C_FLAGS_DEBUG "${_common_options} -g" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_C_FLAGS_MINSIZEREL "${_common_options} -Os -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_C_FLAGS_RELEASE "${_common_options} -O3 -funroll-loops -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "${_common_options} -g -O3 -funroll-loops -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

        set(CMAKE_CXX_FLAGS_DEBUG "${_common_cxx_options} -g" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "${_common_cxx_options} -Os -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "${_common_cxx_options} -O3 -funroll-loops -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${_common_cxx_options} -g -O3 -funroll-loops -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

    elseif(CMAKE_CXX_COMPILER MATCHES "icpc")
        # default flags for intel
        set(CMAKE_C_FLAGS_DEBUG "-xHOST -O0 -g" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_C_FLAGS_MINSIZEREL "-xHOST -Os -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_C_FLAGS_RELEASE "-xHOST -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "-xHOST -g -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

        set(CMAKE_CXX_FLAGS_DEBUG "-xHOST -O0 -g" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-xHOST -Os -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-xHOST -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-xHOST -g -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

    else(CMAKE_COMPILER_IS_GNUCXX)
        message(STATUS "No default CXXFLAGS for your compiler, set them manually")
    endif()

    if (CMAKE_VERSION VERSION_LESS 3.3.0)
        # older versions of cmake don't handle C++11 in FindCUDA - attempt to work around this.
        set(CUDA_NVCC_FLAGS_DEBUG "--std=c++11" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CUDA_NVCC_FLAGS_MINSIZEREL "-std=c++11" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CUDA_NVCC_FLAGS_RELEASE "-std=c++11" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "-std=c++11" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

    endif()

SET(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "First configure has run: CXX_FLAGS have had their defaults changed" FORCE)
endif(NOT PASSED_FIRST_CONFIGURE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
