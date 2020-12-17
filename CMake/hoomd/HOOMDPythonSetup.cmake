set(PYBIND11_PYTHON_VERSION 3)
find_package(pybind11 2.2 CONFIG REQUIRED)

if (pybind11_FOUND)
    find_package_message(pybind11 "Found pybind11: ${pybind11_DIR} ${pybind11_INCLUDE_DIR} (version ${pybind11_VERSION})" "[${pybind11_DIR}][${pybind11_INCLUDE_DIR}]")
endif()

# when the user specifies CMAKE_INSTALL_PREFIX on first configure, install to "hoomd" under that prefix
set(PYTHON_SITE_INSTALL_DIR "hoomd" CACHE PATH
    "Python site-packages directory (relative to CMAKE_INSTALL_PREFIX)")

# when no CMAKE_INSTALL_PREFIX is set, default to python's install prefix
if((CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR "${PYTHON_PREFIX}" STREQUAL "${CMAKE_INSTALL_PREFIX}") AND PYTHON_SITE_PACKAGES)
    string(LENGTH "${PYTHON_PREFIX}" _python_prefix_len)
    string(SUBSTRING "${PYTHON_SITE_PACKAGES}" 0 ${_python_prefix_len} _python_site_package_prefix)
    math(EXPR _shart_char "${_python_prefix_len}+1")
    string(SUBSTRING "${PYTHON_SITE_PACKAGES}" ${_shart_char} -1 _python_site_package_rel)
    string(COMPARE EQUAL "${_python_site_package_prefix}" "${PYTHON_PREFIX}" _prefix_equal)
    if (NOT _prefix_equal)
        message(STATUS "Python site packages (${PYTHON_SITE_PACKAGES}) does not start with python prefix (${PYTHON_PREFIX})")
        message(STATUS "HOOMD may not install to the correct location")
    endif()

    set(CMAKE_INSTALL_PREFIX "${PYTHON_PREFIX}" CACHE PATH "HOOMD installation path" FORCE)
    set(PYTHON_SITE_INSTALL_DIR "${_python_site_package_rel}/hoomd" CACHE PATH
        "Python site-packages directory (relative to CMAKE_INSTALL_PREFIX)" FORCE)
endif()

find_package_message(hoomd_install "Installing hoomd python module to: ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}" "[${CMAKE_INSTALL_PREFIX}][${PYTHON_SITE_INSTALL_DIR}]")
