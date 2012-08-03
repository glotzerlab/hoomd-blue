# Maintainer: joaander

##################################
## find the threads library
find_package(Threads)

include_directories(${PYTHON_INCLUDE_DIR})

# find ZLIB
if (ENABLE_ZLIB)
find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIR})
endif (ENABLE_ZLIB)

if (ENABLE_OCELOT)
find_library(OCELOT_LIBRARY NAMES ocelot)
# override the CUDART library
set (CUDA_CUDART_LIBRARY ${OCELOT_LIBRARY} CACHE STRING "ENABLE_OCELOT forces replacement of CUDART with ocelot" FORCE)

if (OCELOT_LIBRARY)
    mark_as_advanced(OCELOT_LIBRARY)
endif (OCELOT_LIBRARY)

endif (ENABLE_OCELOT)


################################
## Define common libraries used by every target in HOOMD
if (WIN32)
    set(HOOMD_COMMON_LIBS ${PYTHON_LIBRARIES} ${WINSOCK_LIB})
else (WIN32)
    set(BOOST_LIBS ${Boost_THREAD_LIBRARY}
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            ${Boost_SIGNALS_LIBRARY}
            ${Boost_IOSTREAMS_LIBRARY}
            )

    set(BOOST_LIBS ${BOOST_LIBS} ${Boost_PYTHON_LIBRARY})

    if (Boost_SYSTEM_LIBRARY)
        set(BOOST_LIBS ${BOOST_LIBS} ${Boost_SYSTEM_LIBRARY})
    endif (Boost_SYSTEM_LIBRARY)


    # these libraries are needed for MPI
    if (Boost_MPI_LIBRARY AND Boost_SERIALIZATION_LIBRARY)
       set(BOOST_LIBS ${BOOST_LIBS} ${Boost_MPI_LIBRARY} ${Boost_SERIALIZATION_LIBRARY})
    endif (Boost_MPI_LIBRARY AND Boost_SERIALIZATION_LIBRARY)

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
            ${PYTHON_LIBRARIES}
            ${BOOST_LIBS}
            ${CMAKE_THREAD_LIBS_INIT}
            ${ZLIB_LIBRARIES}
            ${ADDITIONAL_LIBS}
            )
endif (WIN32)

if (ENABLE_MPI)
    # Check that we can compile a program against the Boost Libraries
    set(CMAKE_REQUIRED_LIBRARIES ${HOOMD_COMMON_LIBS} ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY})
    set(CMAKE_REQUIRED_FLAGS ${MPI_LINK_FLAGS} ${MPI_COMPILE_FLAGS})
    include(CheckLibraryExists)
    CHECK_LIBRARY_EXISTS("${Boost_MPI_LIBRARY}" exit "" BOOST_CAN_COMPILE)
    if (NOT BOOST_CAN_COMPILE)
       message(WARNING "Cannot link against Boost.MPI. Disabling MPI.")
       set(ENABLE_MPI FALSE CACHE BOOL "Enable the compilation of the MPI communication code" FORCE)
    endif(NOT BOOST_CAN_COMPILE)

    list(APPEND HOOMD_COMMON_LIBS ${MPI_LIBRARY} ${MPI_EXTRA_LIBRARY})
endif (ENABLE_MPI)

if (ENABLE_CUDA)
    list(APPEND HOOMD_COMMON_LIBS ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY})
endif (ENABLE_CUDA)
