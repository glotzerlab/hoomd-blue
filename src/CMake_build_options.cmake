# $Id$
# $URL$
# Maintainer: joaander

#################################
## Optional use of zlib to compress binary output files
option(ENABLE_ZLIB "When set to ON, a gzip compression option for binary output files is available" ON)
if (ENABLE_ZLIB)
    add_definitions(-DENABLE_ZLIB)
endif(ENABLE_ZLIB)

#################################
## Minimal install option
option(ENABLE_MINIMAL_INSTALL "When set to ON, only a minimal set of files needed to to run HOOMD are installed" OFF)

#################################
## Optional static build
## ENABLE_STATIC is an option to control whether HOOMD is built as a statically linked exe or as a python module.
OPTION(ENABLE_STATIC "Link as many libraries as possible statically, cannot be changed after the first run of CMake" ON)
mark_as_advanced(ENABLE_STATIC)
if (ENABLE_STATIC)
    add_definitions(-DENABLE_STATIC)
endif(ENABLE_STATIC)

#################################
## Optional build with double size exclusion list
## LARGE_EXCLUSION_LIST is an option to control whether HOOMD will have extra
## exclusion lists to handle more complex molecules. Has some speed impact and
## thus is not on by default.
OPTION(LARGE_EXCLUSION_LIST "Increase the number of allowed exclusions from 4 to 16 per atom. Only needed for branched molecules with angles and dihedrals." OFF)
if (LARGE_EXCLUSION_LIST)
    add_definitions(-DLARGE_EXCLUSION_LIST)
endif(LARGE_EXCLUSION_LIST)

#################################
## Optional single/double precision build
option(SINGLE_PRECISION "Use single precision math" ON)
if (SINGLE_PRECISION)
    add_definitions (-DSINGLE_PRECISION)
endif (SINGLE_PRECISION)

#####################3
## CUDA related options
option(ENABLE_CUDA "Enable the compilation of the CUDA GPU code" off)
if (ENABLE_CUDA)
    add_definitions (-DENABLE_CUDA)

    # CUDA ARCH settings
    set(CUDA_ARCH 11 CACHE STRING "Target architecture to compile CUDA code for. Valid options are 10, 11, 12, or 13 (currently). They correspond to compute 1.0, 1.1, 1.2, and 1.3 GPU hardware")
    # the arch is going to be passed on a command line: verify it so the user doesn't make any blunders
    set(_cuda_arch_ok FALSE)
    foreach(_valid_cuda_arch 10 11 12 13)
        if (CUDA_ARCH EQUAL ${_valid_cuda_arch})
            set(_cuda_arch_ok TRUE)
        endif (CUDA_ARCH EQUAL ${_valid_cuda_arch})
            endforeach(_valid_cuda_arch)
        if (NOT _cuda_arch_ok)
            message(FATAL_ERROR "Wrong CUDA_ARCH specified. Must be one of 10, 11, 12, or 13")
    endif (NOT _cuda_arch_ok)

    add_definitions(-DCUDA_ARCH=${CUDA_ARCH})
    list(APPEND CUDA_NVCC_FLAGS -arch "sm_${CUDA_ARCH}")
    
    # ULF bug workaround disable option
    option(DISABLE_ULF_WORKAROUND "Set to ON to enable higher performace at the cost of stability on pre C1060 GPUs" off)
    mark_as_advanced(DISABLE_ULF_WORKAROUND)
    if (DISABLE_ULF_WORKAROUND)
        add_definitions (-DDISABLE_ULF_WORKAROUND)
    endif (DISABLE_ULF_WORKAROUND)
endif (ENABLE_CUDA)

#################################
## Optionally enable documentation build
find_package(Doxygen)
if (DOXYGEN_FOUND)
    # get the doxygen version
    exec_program(${DOXYGEN_EXECUTABLE} ${HOOMD_SOURCE_DIR} ARGS --version OUTPUT_VARIABLE DOXYGEN_VERSION)
    STRING(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\.[0-9]+.*" "\\1" DOXYGEN_VERSION_MAJOR "${DOXYGEN_VERSION}")
    STRING(REGEX REPLACE "^[0-9]+\\.([0-9])+\\.[0-9]+.*" "\\1" DOXYGEN_VERSION_MINOR "${DOXYGEN_VERSION}")
    STRING(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" DOXYGEN_VERSION_PATCH "${DOXYGEN_VERSION}")

    if (DOXYGEN_VERSION_MAJOR GREATER 0 AND (DOXYGEN_VERSION_MINOR GREATER 4 AND DOXYGEN_VERSION_PATCH GREATER 5) OR (DOXYGEN_VERSION_MINOR GREATER 5))
        OPTION(ENABLE_DOXYGEN "Enables building of documentation with doxygen" ON)
    else (DOXYGEN_VERSION_MAJOR GREATER 0 AND (DOXYGEN_VERSION_MINOR GREATER 4 AND DOXYGEN_VERSION_PATCH GREATER 5) OR (DOXYGEN_VERSION_MINOR GREATER 5))
        message(STATUS "Doxygen version less than 1.5.5, defaulting ENABLE_DOXYGEN=OFF")
        OPTION(ENABLE_DOXYGEN "Enables building of documentation with doxygen" OFF)
    endif (DOXYGEN_VERSION_MAJOR GREATER 0 AND (DOXYGEN_VERSION_MINOR GREATER 4 AND DOXYGEN_VERSION_PATCH GREATER 5) OR (DOXYGEN_VERSION_MINOR GREATER 5))
endif (DOXYGEN_FOUND)

################################
## thread safe compiling
if(WIN32)
    add_definitions(-D_MT)
elseif(UNIX)
    add_definitions(-D_REENTRANT)
endif(WIN32)
