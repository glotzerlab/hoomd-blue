CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

# Name the plugin project
project(example_plugin LANGUAGES C CXX)

# Find the installed HOOMD
find_package(HOOMD 2.6 REQUIRED)

# Force installation to the HOOMD installation location
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)

message(STATUS "Installing ${PROJECT_NAME} to: ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

# add subdirectories
add_subdirectory(${PROJECT_NAME})
