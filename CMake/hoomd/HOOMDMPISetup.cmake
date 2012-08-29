# Maintainer: jglaser

##################################
## Find MPI
if (ENABLE_MPI)
    # the package is needed
    find_package(MPI REQUIRED)

    # now perform some more in-depth tests of whether the MPI library supports CUDA memory
    if (ENABLE_CUDA AND NOT DEFINED ENABLE_MPI_CUDA)
        if (MPI_LIBRARY MATCHES mpich)
            # find out if this is MVAPICH2
            get_filename_component(_mpi_library_dir ${MPI_LIBRARY} PATH)
            find_program(MPICH2_VERSION
                NAMES mpich2version
                HINTS ${_mpi_library_dir} ${_mpi_library_dir}/../bin
            )
            if (MPICH2_VERSION)
                execute_process(COMMAND ${MPICH2_VERSION}
                                OUTPUT_VARIABLE _output)
                if (_output MATCHES "--enable-cuda")
                    set(MPI_CUDA TRUE)
                    message(STATUS "Found MVAPICH2 with CUDA support.")
                endif()
            endif()
        elseif(MPI_LIBRARY MATCHES libmpi)
            # find out if this OpenMPI
            if (MPI_INCLUDE_PATH)
               foreach(_dir ${MPI_INCLUDE_PATH})
                   if (EXISTS ${_dir}/openmpi/opal_config.h)
                       file( STRINGS ${_dir}/openmpi/opal_config.h
                             _ompi_cuda_support
                             REGEX "#define OMPI_CUDA_SUPPORT[ \t]+([0-9x]+)$"
                           )
                       if (_ompi_cuda_support)
                           string( REGEX REPLACE
                               "#define OMPI_CUDA_SUPPORT[ \t]+"
                               "" _ompi_cuda_support ${_ompi_cuda_support} )
                           if (_ompi_cuda_support MATCHES 1)
                               message(STATUS "Found OpenMPI with CUDA support.")
                               set(MPI_CUDA TRUE)
                           endif()
                       endif()
                    endif()
               endforeach()
            endif()
        endif()

        if (MPI_CUDA)
           message(STATUS "Enabling MPI<->CUDA interoperability.")
           option(ENABLE_MPI_CUDA "Enable MPI<->CUDA interoperability" on)
        else(MPI_CUDA)
           message(STATUS "MPI found, but without CUDA interoperability. Expect slower MPI performance.")
           option(ENABLE_MPI_CUDA "Enable MPI<->CUDA interoperability" off)
        endif(MPI_CUDA)
    endif (ENABLE_CUDA AND NOT DEFINED ENABLE_MPI_CUDA)

    if (ENABLE_MPI)
        # add include directories
        include_directories(${MPI_C_INCLUDE_PATH})
    endif()

endif (ENABLE_MPI)
