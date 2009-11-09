# $Id$
# $URL$
# Maintainer: joaander

#################################
## Setup include directories and file lists for sub directories
include_directories(${HOOMD_SOURCE_DIR}/utils ${HOOMD_SOURCE_DIR}/data_structures ${HOOMD_SOURCE_DIR}/computes
                    ${HOOMD_SOURCE_DIR}/updaters ${HOOMD_SOURCE_DIR}/cuda ${HOOMD_SOURCE_DIR}/analyzers
                    ${HOOMD_SOURCE_DIR}/computes_gpu
                    ${HOOMD_SOURCE_DIR}/updaters_gpu
                    ${HOOMD_SOURCE_DIR}/system
                    ${HOOMD_SOURCE_DIR}/extern
                    ${CUDA_INCLUDE}
                    ${HOOMD_BINARY_DIR}/utils)

# list of all sources in various source directories
file(GLOB UTILS_SRCS ${HOOMD_SOURCE_DIR}/utils/*.cc ${HOOMD_SOURCE_DIR}/utils/*.h)
file(GLOB DATA_STRUCT_SRCS ${HOOMD_SOURCE_DIR}/data_structures/*.cc ${HOOMD_SOURCE_DIR}/data_structures/*.h)
file(GLOB COMPUTES_SRCS ${HOOMD_SOURCE_DIR}/computes/*.cc ${HOOMD_SOURCE_DIR}/computes/*.h)
file(GLOB COMPUTES_GPU_SRCS ${HOOMD_SOURCE_DIR}/computes_gpu/*.cc ${HOOMD_SOURCE_DIR}/computes_gpu/*.h)
file(GLOB UPDATER_SRCS ${HOOMD_SOURCE_DIR}/updaters/*.cc ${HOOMD_SOURCE_DIR}/updaters/*.h)
file(GLOB UPDATER_GPU_SRCS ${HOOMD_SOURCE_DIR}/updaters_gpu/*.cc ${HOOMD_SOURCE_DIR}/updaters_gpu/*.h)
file(GLOB ANALYZER_SRCS ${HOOMD_SOURCE_DIR}/analyzers/*.cc ${HOOMD_SOURCE_DIR}/analyzers/*.h)
file(GLOB EXTERN_SRCS ${HOOMD_SOURCE_DIR}/extern/*.cc ${HOOMD_SOURCE_DIR}/extern/*.h)
file(GLOB SYSTEM_SRCS ${HOOMD_SOURCE_DIR}/system/*.cc ${HOOMD_SOURCE_DIR}/system/*.h)
file(GLOB PYTHON_SRCS ${HOOMD_SOURCE_DIR}/python/hoomd_module.cc)
file(GLOB PAIR_SRCS ${HOOMD_SOURCE_DIR}/pair/*.cc ${HOOMD_SOURCE_DIR}/pair/*.h)
file(GLOB CUDA_SRCS ${HOOMD_SOURCE_DIR}/cuda/*.cu ${HOOMD_SOURCE_DIR}/cuda/*.h ${HOOMD_SOURCE_DIR}/cuda/*.cuh)

# make some convenience variables
set(HOOMD_CPU_SRCS ${UTILS_SRCS} ${DATA_STRUCT_SRCS} ${COMPUTES_SRCS} ${UPDATER_SRCS} ${ANALYZER_SRCS} ${EXTERN_SRCS} ${SYSTEM_SRCS} ${PAIR_SRCS})

set(HOOMD_GPU_SRCS ${COMPUTES_GPU_SRCS} ${UPDATER_GPU_SRCS} ${CUDA_SRCS})

#######################
## Configure the version info header file


# handle linux/mac and windows dates differently
if (NOT WIN32)
    exec_program("date" OUTPUT_VARIABLE COMPILE_DATE)
else(NOT WIN32)
    exec_program("cmd" ARGS "/c date /T" OUTPUT_VARIABLE COMPILE_DATE)
endif (NOT WIN32)

configure_file (${HOOMD_SOURCE_DIR}/utils/HOOMDVersion.h.in ${HOOMD_BINARY_DIR}/utils/HOOMDVersion.h)
