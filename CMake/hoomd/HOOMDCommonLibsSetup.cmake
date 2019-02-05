# Maintainer: joaander

##################################
## find the threads library
find_package(Threads)

include_directories(${HOOMD_PYTHON_INCLUDE_DIR})

################################
## Define common libraries used by every target in HOOMD
set(BOOST_LIBS ${Boost_THREAD_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY}
        ${Boost_SIGNALS_LIBRARY}
        ${Boost_IOSTREAMS_LIBRARY}
        )

string(TOUPPER ${BOOST_PYTHON_COMPONENT} UPPER_BOOST_PYTHON_COMPONENT )
set(BOOST_LIBS ${BOOST_LIBS} ${Boost_${UPPER_BOOST_PYTHON_COMPONENT}_LIBRARY})

string(TOUPPER ${BOOST_NUMPY_COMPONENT} UPPER_BOOST_NUMPY_COMPONENT )
set(BOOST_LIBS ${BOOST_LIBS} ${Boost_${UPPER_BOOST_NUMPY_COMPONENT}_LIBRARY})

if (Boost_SYSTEM_LIBRARY)
    set(BOOST_LIBS ${BOOST_LIBS} ${Boost_SYSTEM_LIBRARY})
endif (Boost_SYSTEM_LIBRARY)


# these libraries are needed for MPI
if(ENABLE_MPI)
set(BOOST_LIBS ${BOOST_LIBS} ${Boost_SERIALIZATION_LIBRARY})
endif(ENABLE_MPI)

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
        ${HOOMD_PYTHON_LIBRARY}
        ${BOOST_LIBS}
        ${CMAKE_THREAD_LIBS_INIT}
        ${ZLIB_LIBRARIES}
        ${ADDITIONAL_LIBS}
        )

if (ENABLE_CUDA)
    list(APPEND HOOMD_COMMON_LIBS ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY})

    if (ENABLE_NVTOOLS)
        list(APPEND HOOMD_COMMON_LIBS ${CUDA_nvToolsExt_LIBRARY})
    endif()
endif (ENABLE_CUDA)

if (ENABLE_MPI)
    list(APPEND HOOMD_COMMON_LIBS ${MPI_CXX_LIBRARIES})
endif (ENABLE_MPI)
