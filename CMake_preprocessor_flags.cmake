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
else(SINGLE_PRECISION)
   add_definitions (-Dkiss_fft_scalar=double)
endif(SINGLE_PRECISION)

if (ENABLE_CUDA)
    add_definitions (-DENABLE_CUDA)

    if (ENABLE_NVTOOLS)
        add_definitions(-DENABLE_NVTOOLS)
    endif()
endif (ENABLE_CUDA)

################################
## thread safe compiling
if(WIN32)
    add_definitions(-D_MT)
elseif(UNIX)
    add_definitions(-D_REENTRANT)
endif(WIN32)

if (ENABLE_MPI)
    add_definitions (-DENABLE_MPI)

    if (ENABLE_MPI_CUDA)
          add_definitions (-DENABLE_MPI_CUDA)
    endif(ENABLE_MPI_CUDA)
endif(ENABLE_MPI)

# define this as a main hoomd build (as opposed to a plugin build)
add_definitions(-DBUILDING_HOOMD)
