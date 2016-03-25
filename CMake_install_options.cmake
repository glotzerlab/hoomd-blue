# Maintainer: joaander

option(ENABLE_EMBED_CUDA "Enable embedding of the CUDA libraries into lib/hoomd" OFF)
mark_as_advanced(ENABLE_EMBED_CUDA)

if (INSTALL_SITE)
    SET(CMAKE_INSTALL_PREFIX ${PYTHON_SYSTEM_SITE} CACHE PATH "Python site installation directory" FORCE)
    message(STATUS "Setting installation site dir: ${CMAKE_INSTALL_PREFIX}")
ENDIF()

set(PYTHON_MODULE_BASE_DIR "hoomd")
