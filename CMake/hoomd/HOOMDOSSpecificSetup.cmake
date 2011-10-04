# Maintainer: joaander

#################################
## On windows: we need winsock
if (WIN32)
    if(CMAKE_CL_64)
    find_library(WINSOCK_LIB WSock32 PATHS 
        $ENV{PLATFORMSDK_DIR}Lib/AMD64 "$ENV{PROGRAMFILES}/Microsoft Platform SDK/Lib/AMD64" 
        "$ENV{PROGRAMFILES}/Microsoft Visual Studio 8/VC/PlatformSDK/Lib/AMD64" 
        DOC "Path to WSock32.lib")
    else(CMAKE_CL_64)
    find_library(WINSOCK_LIB WSock32 PATHS
        $ENV{PLATFORMSDK_DIR}Lib "$ENV{PROGRAMFILES}/Microsoft Platform SDK/Lib" 
        "$ENV{PROGRAMFILES}/Microsoft Visual Studio 8/VC/PlatformSDK/Lib"
        "$ENV{PROGRAMFILES}/Microsoft Platform SDK for Windows Server 2003 R2/Lib"
        DOC "Path to WSock32.lib")
    endif(CMAKE_CL_64)
endif (WIN32)

if (WINSOCK_LIB)
    mark_as_advanced(WINSOCK_LIB)
endif (WINSOCK_LIB)

##################################
## SSE and floating point compilation options
# msvc 2005 doesn't define __SSE__ or __SSE2__, so we define them for it
# this of course assumes that the machine is capable of SSE2.... which almost any windows machine will be these days

if (WIN32)
    add_definitions(-D__SSE__ -D__SSE2__)
endif (WIN32)
