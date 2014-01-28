# Maintainer: joaander

#################################
## Optional use of zlib to compress binary output files (defaults to off on windows)
if (WIN32)
option(ENABLE_ZLIB "When set to ON, a gzip compression option for binary output files is available" OFF)
else (WIN32)
option(ENABLE_ZLIB "When set to ON, a gzip compression option for binary output files is available" ON)
endif (WIN32)

#################################
## Optional static build
## ENABLE_STATIC is an option to control whether HOOMD is built as a statically linked exe or as a python module.
if (WIN32)
OPTION(ENABLE_STATIC "Link as many libraries as possible statically, cannot be changed after the first run of CMake" ON)
else (WIN32)
OPTION(ENABLE_STATIC "Link as many libraries as possible statically, cannot be changed after the first run of CMake" OFF)
endif (WIN32)

mark_as_advanced(ENABLE_STATIC)

#################################
## Optional single/double precision build
option(SINGLE_PRECISION "Use single precision math" ON)

#####################3
## CUDA related options
find_package(CUDA QUIET)
if (CUDA_FOUND)
option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" on)
else (CUDA_FOUND)
option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" off)
endif (CUDA_FOUND)

# disable CUDA if the intel compiler is detected
if (CMAKE_CXX_COMPILER MATCHES "icpc")
    set(ENABLE_CUDA OFF CACHE BOOL "Forced OFF by the use of the intel c++ compiler" FORCE)
endif (CMAKE_CXX_COMPILER MATCHES "icpc")

if (ENABLE_CUDA)
    # optional ocelot emulation mode
    option(ENABLE_OCELOT "Enable ocelot emulation for CUDA GPU code" off)
    if (ENABLE_OCELOT)
        set(CUDA_ARCH "11")
        add_definitions(-DCUDA_ARCH=${CUDA_ARCH})
        list(APPEND CUDA_NVCC_FLAGS -arch "sm_${CUDA_ARCH}")
    endif (ENABLE_OCELOT)

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
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # get the doxygen version
    exec_program(${DOXYGEN_EXECUTABLE} ${HOOMD_SOURCE_DIR} ARGS --version OUTPUT_VARIABLE DOXYGEN_VERSION)

    if (${DOXYGEN_VERSION} VERSION_GREATER 1.5.5)
        OPTION(ENABLE_DOXYGEN "Enables building of documentation with doxygen" ON)
    else (${DOXYGEN_VERSION} VERSION_GREATER 1.5.5)
        message(STATUS "Doxygen version less than 1.5.5, defaulting ENABLE_DOXYGEN=OFF")
        OPTION(ENABLE_DOXYGEN "Enables building of documentation with doxygen" OFF)
    endif (${DOXYGEN_VERSION} VERSION_GREATER 1.5.5)
endif (DOXYGEN_FOUND)

################################
## detect and optionally enable OpenMP

# Apple's openmp is buggy, disable it
if (NOT APPLE)
# needs CMake 2.6.4 or newer
set (_cmake_ver "${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}.${CMAKE_PATCH_VERSION}")
if (_cmake_ver VERSION_GREATER 2.6.3)

find_package(OpenMP)
if (OPENMP_FOUND)
    option(ENABLE_OPENMP "Enable openmp compliation to accelerate CPU code on multi-core machines" OFF)
endif (OPENMP_FOUND)

else (_cmake_ver VERSION_GREATER 2.6.3)
message(STATUS "Upgrade to CMake 2.6.4 or newer to enable OpenMP compilation")
endif (_cmake_ver VERSION_GREATER 2.6.3)
endif (NOT APPLE)

###############################
## install python code into the system site dir, if a system python installation is desired
SET(PYTHON_SITEDIR "" CACHE STRING "System python site-packages directory to install python module code to. If unspecified, install to lib/hoomd/python-module")
if (PYTHON_SITEDIR)
    set(HOOMD_PYTHON_MODULE_DIR ${PYTHON_SITEDIR})
else (PYTHON_SITEDIR)
    set(HOOMD_PYTHON_MODULE_DIR ${LIB_INSTALL_DIR}/python-module)
endif (PYTHON_SITEDIR)
mark_as_advanced(PYTHON_SITEDIR)
