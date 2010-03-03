## $Id$
## $URL$
# Maintainer: joaander

add_executable(hoomd hoomd.cc)
target_link_libraries(hoomd libhoomd ${HOOMD_COMMON_LIBS})
fix_cudart_rpath(hoomd)

# install the executable
install(TARGETS hoomd
        RUNTIME DESTINATION ${LIB_INSTALL_DIR})
    
# install the hoomd runner script
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hoomd_static.in
                ${CMAKE_CURRENT_BINARY_DIR}/hoomd_run_install
                @ONLY
                )

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/hoomd_run_install
    DESTINATION ${BIN_INSTALL_DIR}
    RENAME hoomd
    )
