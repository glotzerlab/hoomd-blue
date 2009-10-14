# $Id$
# $URL$
# Maintainer: joaander

##################################
## find the threads library
find_package(Threads)

#################################
## setup python library and executable
# find the python interpreter, first
find_package(PythonInterp REQUIRED)
# find the python libraries to link to
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_PATH})

################################
## Define common libraries used by every target in HOOMD
if (WIN32)
    set(HOOMD_COMMON_LIBS hoomd_python_module ${PYTHON_LIBRARIES} ${WINSOCK_LIB})
else (WIN32)
    set(BOOST_LIBS ${Boost_THREAD_LIBRARY}
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_PYTHON_LIBRARY}
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            ${Boost_SIGNALS_LIBRARY}
            )
    if (Boost_SYSTEM_LIBRARY)
        set(BOOST_LIBS ${BOOST_LIBS} ${Boost_SYSTEM_LIBRARY})
    endif (Boost_SYSTEM_LIBRARY)

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

    set(HOOMD_COMMON_LIBS
            ${BOOST_LIBS}
            ${CMAKE_THREAD_LIBS_INIT}
            ${PYTHON_LIBRARIES}
            ${ADDITIONAL_LIBS}
            )
endif (WIN32)