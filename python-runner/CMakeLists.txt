## $Id$
## $URL$
# Maintainer: joaander

# handle native python installs separately from the others
if (PYTHON_SITEDIR)
    set(_hoomd_run_input ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_run_native.in)
else (PYTHON_SITEDIR)
    set(_hoomd_run_input ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_run.in)
    
    add_executable(hoomd hoomd.cc)
    target_link_libraries(hoomd libhoomd ${HOOMD_COMMON_LIBS})
    fix_cudart_rpath(hoomd)

    # install the executable
    install(TARGETS hoomd
            RUNTIME DESTINATION ${LIB_INSTALL_DIR})
endif (PYTHON_SITEDIR)
    
# install the hoomd runner script
configure_file(${_hoomd_run_input}
                ${CMAKE_CURRENT_BINARY_DIR}/hoomd_run_install
                @ONLY
                )

if (NOT WIN32)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hoomd_run_install
    DESTINATION ${BIN_INSTALL_DIR}
    RENAME hoomd
    )
endif (NOT WIN32)
