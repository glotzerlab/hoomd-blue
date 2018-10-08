# Maintainer: joaander
# This sets the preprocessor definitions after all build options have been set appropriately

if (SINGLE_PRECISION)
    add_definitions (-DSINGLE_PRECISION)
# disable for now, only support single precision FFT
#else(SINGLE_PRECISION)
#   add_definitions (-Dkiss_fft_scalar=double)
endif(SINGLE_PRECISION)

if (ENABLE_CUDA)
    add_definitions (-DENABLE_CUDA)

    if (ENABLE_NVTOOLS)
        add_definitions(-DENABLE_NVTOOLS)
    endif()

    if(ALWAYS_USE_MANAGED_MEMORY)
        add_definitions(-DALWAYS_USE_MANAGED_MEMORY)
    endif()
endif (ENABLE_CUDA)

################################
## thread safe compiling
add_definitions(-D_REENTRANT)

if (ENABLE_MPI)
    add_definitions (-DENABLE_MPI)

    if (ENABLE_MPI_CUDA)
          add_definitions (-DENABLE_MPI_CUDA)
    endif(ENABLE_MPI_CUDA)
endif(ENABLE_MPI)

# define Eigen should be MPL 2 only
add_definitions(-DEIGEN_MPL2_ONLY)

# export cusolver availability
if (CUSOLVER_AVAILABLE)
    add_definitions(-DCUSOLVER_AVAILABLE)
endif()

# export TBB compile flag
if (ENABLE_TBB)
    add_definitions(-DENABLE_TBB)
endif()
