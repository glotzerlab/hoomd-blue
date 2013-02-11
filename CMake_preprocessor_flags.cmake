# Maintainer: joaander
# This sets the preprocessor definitinos after all build options have been set appropriately

if (ENABLE_ZLIB)
    add_definitions(-DENABLE_ZLIB)
endif(ENABLE_ZLIB)

if (ENABLE_STATIC)
    add_definitions(-DENABLE_STATIC)
endif(ENABLE_STATIC)

if (SINGLE_PRECISION)
    add_definitions (-DSINGLE_PRECISION)
endif (SINGLE_PRECISION)

if (ENABLE_CUDA)
    add_definitions (-DENABLE_CUDA)
endif (ENABLE_CUDA)

################################
## thread safe compiling
if(WIN32)
    add_definitions(-D_MT)
elseif(UNIX)
    add_definitions(-D_REENTRANT)
endif(WIN32)

if (ENABLE_OPENMP)
    add_definitions (-DENABLE_OPENMP)
endif (ENABLE_OPENMP)

if (ENABLE_MPI)
    add_definitions (-DENABLE_MPI)

    if (ENABLE_MPI_CUDA)
          add_definitions (-DENABLE_MPI_CUDA)
    endif(ENABLE_MPI_CUDA)
endif(ENABLE_MPI)
