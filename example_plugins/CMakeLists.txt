# CMakeLists.txt template for building a plugin as an external component.

CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

# Name the plugin project
project(example_plugins LANGUAGES C CXX)

# Find the installed HOOMD
find_package(HOOMD 3.0.0 REQUIRED)

message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

# Force installation to the HOOMD installation location
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)

# Add subdirectories
add_subdirectory(updater_plugin)
add_subdirectory(pair_plugin)
