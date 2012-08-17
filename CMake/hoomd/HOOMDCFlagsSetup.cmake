# Maintainer: joaander

#################################
## Setup default CXXFLAGS
if(NOT PASSED_FIRST_CONFIGURE)
    message(STATUS "Overriding CMake's default CFLAGS (this should appear only once)")

    # default build type is Release when compiling make files
    if(NOT CMAKE_BUILD_TYPE AND NOT HONOR_GENTOO_FLAGS)
       if(${CMAKE_GENERATOR} STREQUAL "Xcode")
       
       else(${CMAKE_GENERATOR} STREQUAL "Xcode")
            set(CMAKE_BUILD_TYPE "Release" CACHE STRING  "Build type: options are None, Release, Debug, RelWithDebInfo" FORCE)
        endif(${CMAKE_GENERATOR} STREQUAL "Xcode")
    endif(NOT CMAKE_BUILD_TYPE AND NOT HONOR_GENTOO_FLAGS)

    if(CMAKE_COMPILER_IS_GNUCXX)
        # special handling to honor gentoo flags
        if (HONOR_GENTOO_FLAGS)
        set(CMAKE_CXX_FLAGS_DEBUG "-Wall" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-Wall -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-DNDEBUG -Wall" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)
        
        else (HONOR_GENTOO_FLAGS)

        # default flags for g++
        set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wno-unknown-pragmas" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -Wall -Wno-unknown-pragmas -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -DNDEBUG -Wall -Wno-unknown-pragmas" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -funroll-loops -DNDEBUG -Wall -Wno-unknown-pragmas" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)
    
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
        set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG" CACHE STRING "Flags used by the compiler during minimum size release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O3 -DNDEBUG" CACHE STRING "Flags used by the compiler during release builds with debug info." FORCE)

    else(CMAKE_COMPILER_IS_GNUCXX)
        message(STATUS "No default CXXFLAGS for your compiler, set them manually")
    endif(CMAKE_COMPILER_IS_GNUCXX)
        
SET(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "First configure has run: CXX_FLAGS have had their defaults changed" FORCE)
endif(NOT PASSED_FIRST_CONFIGURE)

# add openmp compile option to the flags
if (NOT APPLE)
find_package(OpenMP)
if (ENABLE_OPENMP AND OPENMP_FOUND)
    # these changes unfortunately don't make it into the cache, but the user doesn't really need to see it, they can control it with the ENABLE_OPENMP option
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
endif (ENABLE_OPENMP AND OPENMP_FOUND)
endif (NOT APPLE)

# disable crazy windows warnings
if (WIN32)
add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
endif (WIN32)
