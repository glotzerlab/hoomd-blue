# $Id$
# $URL$
# Maintainer: joaander

#################################
## Setup default CXXFLAGS
if(NOT PASSED_FIRST_CONFIGURE)
    message(STATUS "Overriding CMake's default CFLAGS (this should appear only once)")
    
    if(CMAKE_COMPILER_IS_GNUCXX)
        # special handling to honor gentoo flags
        if (HONOR_GENTOO_FLAGS)
        set(CMAKE_CXX_FLAGS_DEBUG "-Wall" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-Wall -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)
        
        else (HONOR_GENTOO_FLAGS)

        # default flags for g++
        # default build type is Release when compiling make files
        if(NOT CMAKE_BUILD_TYPE)
            if(${CMAKE_GENERATOR} STREQUAL "Xcode")
            else(${CMAKE_GENERATOR} STREQUAL "Xcode")
                set(CMAKE_BUILD_TYPE "Release" CACHE STRING  "Build type: options are None, Release, Debug, RelWithDebInfo" FORCE)
            endif(${CMAKE_GENERATOR} STREQUAL "Xcode")
        endif(NOT CMAKE_BUILD_TYPE)

        set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -Wall -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O -funroll-loops -DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)
    
        endif (HONOR_GENTOO_FLAGS)
    elseif(MSVC80)
        # default flags for visual studio 2005
        if (CMAKE_CL_64)
            set(SSE_OPT "")
        else (CMAKE_CL_64)
            set(SSE_OPT "/arch:SSE2")
        endif (CMAKE_CL_64)
        
        set(CMAKE_CXX_FLAGS "/DWIN32 /D_WINDOWS /W3 /Zm1000 /EHa /GR /MP" CACHE STRING "Flags used by all build types." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "${SSE_OPT} /Oi /Ot /Oy /fp:fast /MD /Ox /Ob2 /D NDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${SSE_OPT} /Oi /Ot /Oy /fp:fast /MD /Zi /Ox /Ob1 /D NDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "${SSE_OPT} /Oi /Ot /Oy /fp:fast /MD /O1 /Ob1 /D NDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
    
    else(CMAKE_COMPILER_IS_GNUCXX)
        message(SEND_ERROR "No default CXXFLAGS for your compiler, set them manually")
    endif(CMAKE_COMPILER_IS_GNUCXX)
        
SET(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "First configure has run: CXX_FLAGS have had their defaults changed" FORCE)
endif(NOT PASSED_FIRST_CONFIGURE)


if (CMAKE_COMPILER_IS_GNUCXX)
    set(CUDA_ADDITIONAL_OPTIONS "-Xcompiler;-Wno-strict-aliasing")
else (CMAKE_COMPILER_IS_GNUCXX)
    set(CUDA_ADDITIONAL_OPTIONS "")
endif (CMAKE_COMPILER_IS_GNUCXX)

