# Maintainer: joaander

#################################
## Boost is a required library

# setup the boost static linkage
set(Boost_USE_STATIC_LIBS "OFF")

# setup some additional boost versions so that the newest versions of boost will be found
set(Boost_ADDITIONAL_VERSIONS "1.60.0;1.53.0;1.52.0;1.51.0;1.50.0;1.49.0;1.48.0;1.47.0;1.46;1.46.0;1.45.1;1.45.0;1.45;1.44.1;1.44.0;1.44;1.43.1;1.43.0;1.43;1.42.1;1.42.0;1.42;1.41.0;1.41;1.41;1.40.0;1.40;1.39.0;1.39;1.38.0;1.38")

# When BOOST_ROOT is specified, make sure that we find the one the user intends
if ((BOOST_ROOT OR NOT $ENV{BOOST_ROOT} STREQUAL "") OR NOT $ENV{BOOSTROOT} STREQUAL "" OR NOT $ENV{Boost_DIR} STREQUAL "")
    set(Boost_NO_SYSTEM_PATHS ON)
endif()

set(REQUIRED_BOOST_COMPONENTS signals unit_test_framework serialization)

message(STATUS "First attempt to find boost, it's OK if it fails")
# first, see if we can get any supported version of Boost
find_package(Boost 1.32.0 COMPONENTS ${REQUIRED_BOOST_COMPONENTS})

# if we get boost 1.60 or greater, we need to get the timer, chrono, and system libraries too
if (Boost_MINOR_VERSION GREATER 59)
list(APPEND REQUIRED_BOOST_COMPONENTS "timer" "chrono" "system")
find_package(Boost 1.32.0 COMPONENTS REQUIRED ${REQUIRED_BOOST_COMPONENTS})
endif ()

# add include directories
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

# hide variables the user doesn't need to see
mark_as_advanced(Boost_LIB_DIAGNOSTIC_DEFINITIONS)
mark_as_advanced(Boost_DIR)
