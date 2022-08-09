option (ENABLE_MPI "Enable the compilation of the MPI communication code" off)

##################################
## Find MPI
if (ENABLE_MPI)
    # the package is needed
    find_package(MPI REQUIRED)
    find_package(cereal CONFIG)
    if (cereal_FOUND)
        find_package_message(cereal "Found cereal: ${cereal_DIR}" "[${cereal_DIR}]")

        if (NOT TARGET cereal::cereal AND TARGET cereal)
            message(STATUS "Found cereal target, adding cereal::cereal alias.")
            add_library(cereal::cereal ALIAS cereal)
        endif()
    else()
        # work around missing ceralConfig.cmake (common on Ubuntu 20.04)
        find_path(cereal_INCLUDE_DIR NAMES cereal/cereal.hpp
            PATHS ${CMAKE_INSTALL_PREFIX}/include)
        add_library(cereal::cereal INTERFACE IMPORTED)
        set_target_properties(cereal::cereal PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${cereal_INCLUDE_DIR}")
        find_package_message(cereal "Could not find cereal by config file, falling back to ${cereal_INCLUDE_DIR}" "[${cereal_INCLUDE_DIR}]")
    endif()

    # Work around broken cereal::cereal target (common on Ubuntu 22.04)
    get_target_property(_cereal_include cereal::cereal INTERFACE_INCLUDE_DIRECTORIES)
    if (_cereal_include STREQUAL "/include")
        find_path(cereal_INCLUDE_DIR NAMES cereal/cereal.hpp
            PATHS ${CMAKE_INSTALL_PREFIX}/include)
        set_target_properties(cereal::cereal PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${cereal_INCLUDE_DIR}")
        find_package_message(cereal "Fixing broken cereal::cereal target with ${cereal_INCLUDE_DIR}" "[${cereal_INCLUDE_DIR}]")
    endif()

    mark_as_advanced(MPI_EXTRA_LIBRARY)
    mark_as_advanced(MPI_LIBRARY)
    mark_as_advanced(OMPI_INFO)

    # now perform some more in-depth tests of whether the MPI library supports CUDA memory
    if (ENABLE_HIP AND NOT DEFINED ENABLE_MPI_CUDA)
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
    endif (ENABLE_HIP AND NOT DEFINED ENABLE_MPI_CUDA)

if (ENABLE_HIP)
    string(REPLACE "-pthread" "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<STREQUAL:${HIP_PLATFORM},nvcc>>:-Xcompiler>;-pthread"
      _MPI_C_COMPILE_OPTIONS "${MPI_C_COMPILE_OPTIONS}")
    set_property(TARGET MPI::MPI_C PROPERTY INTERFACE_COMPILE_OPTIONS "${_MPI_C_COMPILE_OPTIONS}")
    unset(_MPI_C_COMPILE_OPTIONS)

    string(REPLACE "-pthread" "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<STREQUAL:${HIP_PLATFORM},nvcc>>:-Xcompiler>;-pthread"
      _MPI_CXX_COMPILE_OPTIONS "${MPI_CXX_COMPILE_OPTIONS}")
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS "${_MPI_CXX_COMPILE_OPTIONS}")
    unset(_MPI_CXX_COMPILE_OPTIONS)
endif()

endif (ENABLE_MPI)
