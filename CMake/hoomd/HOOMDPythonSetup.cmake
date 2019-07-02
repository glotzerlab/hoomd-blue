# use pybind11's tools to find python and create python modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/hoomd/extern/pybind/tools)

# for installed hoomd with external plugins
if (HOOMD_INCLUDE_DIR)
    list(APPEND CMAKE_MODULE_PATH ${HOOMD_INCLUDE_DIR}/hoomd/extern/pybind/tools)
endif()

set(PYBIND11_PYTHON_VERSION 3)

# trick pybind11 tools to allow us to manage C++ flags
# cmake ignores this in 2.8, but when pybind11 sees this
# it will not override hoomd's cflags
set(CMAKE_CXX_STANDARD 11)

# hoomd's cflags setup script will take care of proper cxx flags settings

include(pybind11Tools)

if (PYTHON_VERSION_MAJOR LESS 3)
    message(SEND_ERROR "HOOMD requires Python 3")
endif()
