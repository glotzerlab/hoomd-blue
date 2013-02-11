# Maintainer: joaander

#################################
## Setup include directories and file lists for sub directories
include_directories(${HOOMD_SOURCE_DIR}/libhoomd/utils
                    ${HOOMD_SOURCE_DIR}/libhoomd/data_structures
                    ${HOOMD_SOURCE_DIR}/libhoomd/computes
                    ${HOOMD_SOURCE_DIR}/libhoomd/updaters
                    ${HOOMD_SOURCE_DIR}/libhoomd/cuda
                    ${HOOMD_SOURCE_DIR}/libhoomd/analyzers
                    ${HOOMD_SOURCE_DIR}/libhoomd/potentials
                    ${HOOMD_SOURCE_DIR}/libhoomd/computes_gpu
                    ${HOOMD_SOURCE_DIR}/libhoomd/updaters_gpu
                    ${HOOMD_SOURCE_DIR}/libhoomd/system
                    ${HOOMD_SOURCE_DIR}/libhoomd/extern
                    ${HOOMD_SOURCE_DIR}/libhoomd/communication
                    ${CUDA_INCLUDE}
                    ${HOOMD_BINARY_DIR}/include)

#######################
## Configure the version info header file

# handle linux/mac and windows dates differently
if (NOT WIN32)
    exec_program("date" OUTPUT_VARIABLE COMPILE_DATE)
else(NOT WIN32)
    exec_program("cmd" ARGS "/c date /T" OUTPUT_VARIABLE COMPILE_DATE)
endif (NOT WIN32)
