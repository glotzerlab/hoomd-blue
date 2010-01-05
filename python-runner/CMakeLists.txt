## $Id$
## $URL$
# Maintainer: joaander

if (NOT ENABLE_STATIC)
    set(_hoomd_runner_in ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_dynamic.in)
endif (NOT ENABLE_STATIC)

# need to build a special python executable w/ static libraries
if (ENABLE_STATIC)
    add_executable(hoomd hoomd.cc)
    target_link_libraries(hoomd libhoomd ${HOOMD_COMMON_LIBS})
    fix_cudart_rpath(hoomd)

    # install the executable
    install(TARGETS hoomd
           RUNTIME DESTINATION ${LIB_INSTALL_DIR})
    
    set(_hoomd_runner_in ${CMAKE_CURRENT_SOURCE_DIR}/hoomd_static.in)
endif (ENABLE_STATIC)

# install the hoomd runner script
configure_file(${_hoomd_runner_in}
                ${CMAKE_CURRENT_BINARY_DIR}/hoomd_runner
                @ONLY
                )

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hoomd_dynamic
    DESTINATION ${BIN_INSTALL_DIR}
    RENAME hoomd
    )
