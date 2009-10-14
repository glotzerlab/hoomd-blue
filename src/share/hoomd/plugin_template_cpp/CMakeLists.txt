project(plugin_template)
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

# plugins must be built as shared libraries
set(BUILD_SHARED_LIBS on)

# collect all the sources together
add_subdirectory(cppmodule)
add_subdirectory(pymodule)
