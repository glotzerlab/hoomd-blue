# Maintainer: joaander

#################################
## Setup include directories and file lists for sub directories
include_directories(${HOOMD_SOURCE_DIR}
                    ${HOOMD_SOURCE_DIR}/hoomd/extern/cereal/include
                    ${CUDA_INCLUDE}
                    ${HOOMD_BINARY_DIR}/hoomd/include)

#######################
## Get the compile date
exec_program("date +%x" OUTPUT_VARIABLE COMPILE_DATE)
