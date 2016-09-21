# Maintainer: joaander

#################################
## Optional static build
## ENABLE_STATIC is an option to control whether HOOMD is built as a statically linked exe or as a python module.
OPTION(ENABLE_STATIC "Link as many libraries as possible statically, cannot be changed after the first run of CMake" OFF)

mark_as_advanced(ENABLE_STATIC)

#################################
## Optional single/double precision build
option(SINGLE_PRECISION "Use single precision math" OFF)

option(ENABLE_HPMC_MIXED_PRECISION "Enable mixed precision computations in HPMC" ON)
if (ENABLE_HPMC_MIXED_PRECISION)
    add_definitions(-DENABLE_HPMC_MIXED_PRECISION)
endif()

#####################3
## CUDA related options
find_package(CUDA QUIET)
if (CUDA_FOUND)
option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" on)
else (CUDA_FOUND)
option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" off)
endif (CUDA_FOUND)

if (ENABLE_CUDA)
    option(ENABLE_NVTOOLS "Enable NVTools profiler integration" off)
endif (ENABLE_CUDA)

############################
## MPI related options
find_package(MPI)
if (MPI_FOUND OR MPI_C_FOUND OR MPI_CXX_FOUND)
option(ENABLE_MPI "Enable the compilation of the MPI communication code" on)
else ()
option (ENABLE_MPI "Enable the compilation of the MPI communication code" off)
endif ()

#################################
## Optionally enable documentation build
OPTION(ENABLE_DOXYGEN "Enables building of documentation with doxygen" OFF)
if (ENABLE_DOXYGEN)
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        # get the doxygen version
        exec_program(${DOXYGEN_EXECUTABLE} ${HOOMD_SOURCE_DIR} ARGS --version OUTPUT_VARIABLE DOXYGEN_VERSION)

        if (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
        else (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
            message(STATUS "Doxygen version less than 1.8.5, documentation may not build correctly")
        endif (${DOXYGEN_VERSION} VERSION_GREATER 1.8.4)
    endif ()
endif ()

option(COPY_HEADERS "Copy the headers into the build directories for plugins" off)
