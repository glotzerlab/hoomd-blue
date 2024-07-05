# CMakeLists.txt template for building a plugin as an external component.

CMAKE_MINIMUM_REQUIRED(VERSION 3.15...3.27 FATAL_ERROR)

# Search for Python and other libraries in unix-like locations first and frameworks last.
# This allows FindPython to find virtual environment Pythons before a homebrew or system Pythons.
set(CMAKE_FIND_FRAMEWORK LAST)

# Name the plugin project
project(example_plugins LANGUAGES C CXX)

# Find the installed HOOMD
find_package(HOOMD 4.0.0 REQUIRED)

message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

# Force installation to the HOOMD installation location
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)

# Enable compiler warnings on gcc and clang (common compilers used by developers)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wconversion -Wno-sign-conversion -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-result")
endif()

# Add subdirectories
add_subdirectory(updater_plugin)
add_subdirectory(pair_plugin)
add_subdirectory(shape_plugin)
