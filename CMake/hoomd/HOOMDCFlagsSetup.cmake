# Maintainer: joaander

#################################
## Setup default CXXFLAGS
if(NOT PASSED_FIRST_CONFIGURE)
    message(STATUS "Overriding CMake's default CFLAGS (this should appear only once)")

    ## Allow GCC_ARCH flag to set the -march field
    if("$ENV{GCC_ARCH}" STREQUAL "")
        set(GCC_ARCH "native")
        message(STATUS "GCC_ARCH env var not set, setting -march to ${GCC_ARCH}")
    else()
        set(GCC_ARCH $ENV{GCC_ARCH})
        message(STATUS "Found GCC_ARCH env var, setting -march to ${GCC_ARCH}")
    endif()


    # default build type is Release when compiling make files
    if(NOT CMAKE_BUILD_TYPE AND NOT HONOR_GENTOO_FLAGS)
       if(${CMAKE_GENERATOR} STREQUAL "Xcode")

       else(${CMAKE_GENERATOR} STREQUAL "Xcode")
            set(CMAKE_BUILD_TYPE "Release" CACHE STRING  "Build type: options are None, Release, Debug, RelWithDebInfo" FORCE)
        endif(${CMAKE_GENERATOR} STREQUAL "Xcode")
    endif(NOT CMAKE_BUILD_TYPE AND NOT HONOR_GENTOO_FLAGS)

    if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        # special handling to honor gentoo flags
        if (HONOR_GENTOO_FLAGS)
        set(CMAKE_CXX_FLAGS_DEBUG "-Wall" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-Wall -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

        else (HONOR_GENTOO_FLAGS)

        # default flags for g++
        set(CMAKE_CXX_FLAGS_DEBUG "-march=${GCC_ARCH} -g -Wall -Wno-unknown-pragmas" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-march=${GCC_ARCH} -Os -Wall -Wno-unknown-pragmas -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-march=${GCC_ARCH} -O3 -funroll-loops -DNDEBUG -Wall -Wno-unknown-pragmas" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-march=${GCC_ARCH} -g -O3 -funroll-loops -DNDEBUG -Wall -Wno-unknown-pragmas" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

        endif (HONOR_GENTOO_FLAGS)
    elseif(MSVC80)
        # default flags for visual studio 2005
        if (CMAKE_CL_64)
            set(SSE_OPT "")
        else (CMAKE_CL_64)
            set(SSE_OPT "/arch:SSE2")
        endif (CMAKE_CL_64)

        set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS /W3 /Zm1000 /EHs /GR /MP" CACHE STRING "Flags used by all build types." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "${SSE_OPT} /Oi /Ot /Oy /fp:fast /MD /Ox /Ob2 /D NDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${SSE_OPT} /Oi /Ot /Oy /fp:fast /MD /Zi /Ox /Ob1 /D NDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "${SSE_OPT} /Oi /Ot /Oy /fp:fast /MD /O1 /Ob1 /D NDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)

    elseif(CMAKE_CXX_COMPILER MATCHES "icpc")
        # default flags for intel
        set(CMAKE_CXX_FLAGS_DEBUG "-xHOST -O0 -g" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-xHOST -Os -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-xHOST -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-xHOST -g -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

    else(CMAKE_COMPILER_IS_GNUCXX)
        message(STATUS "No default CXXFLAGS for your compiler, set them manually")
    endif(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")

SET(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "First configure has run: CXX_FLAGS have had their defaults changed" FORCE)
endif(NOT PASSED_FIRST_CONFIGURE)

# disable crazy windows warnings
if (WIN32)
add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
endif (WIN32)
