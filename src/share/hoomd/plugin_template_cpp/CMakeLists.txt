project(plugin_template)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})
find_package(HOOMD)

# plugins must be built as shared libraries
set(BUILD_SHARED_LIBS on)

# collect all the sources together
add_subdirectory(cppmodule)
add_subdirectory(pymodule)

# ovveride the CMAKE_INSTALL_PREFIX on the first run to ${HOOMD_ROOT}
# ${HOOMD_ROOT} will likely be the most common installation directory, but users may want to override
if (NOT _first_run)
    if (HOOMD_FOUND)
        set(CMAKE_INSTALL_PREFIX ${HOOMD_ROOT} CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
        SET(_first_run ON CACHE INTERNAL "First configure has run: CMAKE_INSTALL_PREFIX has had its defaults changed" FORCE)
    endif (HOOMD_FOUND)
endif (NOT _first_run)
