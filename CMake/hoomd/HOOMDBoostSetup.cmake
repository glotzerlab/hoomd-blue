# Maintainer: joaander

#################################
## Boost is a required library

# setup the boost static linkage
if(ENABLE_STATIC)
    set(Boost_USE_STATIC_LIBS "ON")
    add_definitions(-DBOOST_PYTHON_STATIC_LIB)
else(ENABLE_STATIC)
    if(WIN32)
        message(FATAL_ERROR "Dynamically linking boost to HOOMD on windows is a hopeless cause. If 
            you really want this feature, make it work yourself")
    endif(WIN32)
    set(Boost_USE_STATIC_LIBS "OFF")
endif(ENABLE_STATIC)

# setup some additional boost versions so that the newest versions of boost will be found
set(Boost_ADDITIONAL_VERSIONS "1.53.0;1.52.0;1.51.0;1.50.0;1.49.0;1.48.0;1.47.0;1.46;1.46.0;1.45.1;1.45.0;1.45;1.44.1;1.44.0;1.44;1.43.1;1.43.0;1.43;1.42.1;1.42.0;1.42;1.41.0;1.41;1.41;1.40.0;1.40;1.39.0;1.39;1.38.0;1.38")

# When BOOST_ROOT is specified, make sure that we find the one the user intends
if ((BOOST_ROOT OR NOT $ENV{BOOST_ROOT} STREQUAL "") OR NOT $ENV{BOOSTROOT} STREQUAL "" OR NOT $ENV{Boost_DIR} STREQUAL "")
    set(Boost_NO_SYSTEM_PATHS ON)
endif()

# if (PYTHON_VERSION VERSION_GREATER 3)
# set(BOOST_PYTHON_COMPONENT "python3")
# else()
set(BOOST_PYTHON_COMPONENT "python")
# endif()

set(REQUIRED_BOOST_COMPONENTS thread filesystem ${BOOST_PYTHON_COMPONENT} signals program_options unit_test_framework iostreams)

# first, see if we can get any supported version of Boost
find_package(Boost 1.32.0 COMPONENTS REQUIRED ${REQUIRED_BOOST_COMPONENTS})

# if python is not found, try python-X.Y (gentoo style)
if (NOT Boost_PYTHON_FOUND)
message(STATUS "Python ${BOOST_PYTHON_COMPONENT} not found, trying python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
list(REMOVE_ITEM REQUIRED_BOOST_COMPONENTS ${BOOST_PYTHON_COMPONENT})
list(APPEND REQUIRED_BOOST_COMPONENTS "python-${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
endif()

# if we get boost 1.35 or greator, we need to get the system library too
if (Boost_MINOR_VERSION GREATER 34)
list(APPEND REQUIRED_BOOST_COMPONENTS "system")
endif ()

find_package(Boost 1.32.0 COMPONENTS REQUIRED ${REQUIRED_BOOST_COMPONENTS})

# add include directories
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

if (WIN32)
# link directories are needed on windows
link_directories(${Boost_LIBRARY_DIRS})

# the user needs to see if the boost auto-linking is working on windows
# Disabled because it is getting annoying. Renable if you need to debug
# add_definitions(-DBOOST_LIB_DIAGNOSTIC)

# also enable stdcall boost::bind support on windows for CUDA runtime API calls
# but it isn't needed in 64 bit
if (NOT CMAKE_CL_64)
    add_definitions(-DBOOST_BIND_ENABLE_STDCALL)
endif (NOT CMAKE_CL_64)

# hide the diagnostic lib definitions variable
mark_as_advanced(Boost_LIB_DIAGNOSTIC_DEFINITIONS)

endif (WIN32)
