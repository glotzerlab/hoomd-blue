# Maintainer: joaander

# handle native python installs separately from the others
if (PYTHON_SITEDIR)
    # install the hoomd runner script to transparently process hoomd into python
    install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd
            DESTINATION ${BIN_INSTALL_DIR}
            )
    if (ENABLE_MPI)
        message(FATAL_ERROR "Cannot use Python site directory install with MPI support.")
    endif (ENABLE_MPI)
else (PYTHON_SITEDIR)
    # setup and install the hoomd executable
    set(_hoomd_run_input ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_run.in)
    
    add_executable(hoomd hoomd.cc)
    target_link_libraries(hoomd libhoomd ${HOOMD_COMMON_LIBS})
    fix_cudart_rpath(hoomd)

    # set the rpath so that the exe can find libs stashed in lib/hoomd/python-module
    set_target_properties(hoomd PROPERTIES INSTALL_RPATH "\\\$ORIGIN/../lib/hoomd/python-module")

    if (ENABLE_MPI)
       # if we are compiling with MPI support built in, set appropriate
       # compiler/linker flags
       if(MPI_COMPILE_FLAGS)
           set_target_properties(hoomd PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
       endif(MPI_COMPILE_FLAGS)
       if(MPI_LINK_FLAGS)
           set_target_properties(hoomd PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
       endif(MPI_LINK_FLAGS)

       # link to required boost libraries
es
    target_link_libraries(hoomd ${Boost_LIBRARIES} ${MPI_LIBRARY})
    endif(ENABLE_MPI)


    # install the executable
    install(TARGETS hoomd
            RUNTIME DESTINATION ${BIN_INSTALL_DIR})
endif (PYTHON_SITEDIR)
    
