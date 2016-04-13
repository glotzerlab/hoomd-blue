# Maintainer: joaander

#################################
## Setup include directories and file lists for sub directories
include_directories(${HOOMD_SOURCE_DIR}
                    ${CUDA_INCLUDE}
                    ${HOOMD_BINARY_DIR}/include)

if (ENABLE_MPI)
    # include dfftlib headers
    include_directories(${HOOMD_SOURCE_DIR}/libhoomd/extern/dfftlib/src)
    include_directories("${PROJECT_BINARY_DIR}")
endif(ENABLE_MPI)

#######################
## Get the compile date
exec_program("date +%x" OUTPUT_VARIABLE COMPILE_DATE)
