# use pybind11's tools to find python and create python modules
# TODO: add search path for make installed hoomd as well
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/hoomd/extern/pybind/tools)

include(pybind11Tools)
