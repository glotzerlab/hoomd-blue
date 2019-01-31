# Maintainer: jglaser

##################################
## Find MPI
if (ENABLE_MPI)
    # the package is needed
    find_package(MPI REQUIRED)

    mark_as_advanced(MPI_EXTRA_LIBRARY)
    mark_as_advanced(MPI_LIBRARY)
    mark_as_advanced(OMPI_INFO)

    # now perform some more in-depth tests of whether the MPI library supports CUDA memory
    if (ENABLE_CUDA AND NOT DEFINED ENABLE_MPI_CUDA)
        if (MPI_LIBRARY MATCHES mpich)
            # find out if this is MVAPICH2
            get_filename_component(_mpi_library_dir ${MPI_LIBRARY} PATH)
            find_program(MPICH2_VERSION
                NAMES mpichversion mpich2version
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
            # find out if this is OpenMPI
            get_filename_component(_mpi_library_dir ${MPI_LIBRARY} PATH)
            find_program(OMPI_INFO
                NAMES ompi_info
                HINTS ${_mpi_library_dir} ${_mpi_library_dir}/../bin
            )
            if (OMPI_INFO)
                execute_process(COMMAND ${OMPI_INFO}
                                OUTPUT_VARIABLE _output)
                if (_output MATCHES "smcuda")
                    set(MPI_CUDA TRUE)
                    message(STATUS "Found OpenMPI with CUDA support.")
                endif()
            endif()
        endif()

        if (MPI_CUDA)
           option(ENABLE_MPI_CUDA "Enable MPI<->CUDA interoperability" off)
        else(MPI_CUDA)
           option(ENABLE_MPI_CUDA "Enable MPI<->CUDA interoperability" off)
        endif(MPI_CUDA)
    endif (ENABLE_CUDA AND NOT DEFINED ENABLE_MPI_CUDA)

    if (ENABLE_MPI)
        # add include directories
        include_directories(${MPI_C_INCLUDE_PATH})
    endif()

endif (ENABLE_MPI)
