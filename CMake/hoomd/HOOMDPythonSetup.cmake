# use pybind11's tools to find python and create python modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/hoomd/extern/pybind/tools)

# for installed hoomd with external plugins
if (HOOMD_INCLUDE_DIR)
    list(APPEND CMAKE_MODULE_PATH ${HOOMD_INCLUDE_DIR}/hoomd/extern/pybind/tools)
endif()

include(pybind11Tools)

set(PYTHON_MODULE_BASE_DIR "hoomd")
