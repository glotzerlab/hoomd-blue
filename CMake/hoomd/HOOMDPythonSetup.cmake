# use pybind11's tools to find python and create python modules
# TODO: add search path for make installed hoomd as well
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/hoomd/extern/pybind/tools)

# trick pybind11 tools to allow us to manage C++ flags
# cmake ignores this in 2.8, but when pybind11 sees this
# it will not override hoomd's cflags
set(CMAKE_CXX_STANDARD 11)

# hoomd's clfags setup script will take care of proper cxx flags settings

include(pybind11Tools)
