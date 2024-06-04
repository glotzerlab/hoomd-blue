set(Python_FIND_UNVERSIONED_NAMES "FIRST")
find_package(Python REQUIRED COMPONENTS Interpreter Development)

if (Python_FOUND)
    execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import sys; print(sys.prefix, end='')"
                    RESULT_VARIABLE _success
                    OUTPUT_VARIABLE _python_prefix
                    ERROR_VARIABLE _error)

    if(NOT _success MATCHES 0)
        message(FATAL_ERROR "Unable to determine python prefix: \n${_error}")
    endif()
endif()

find_package(pybind11 2.12 CONFIG REQUIRED)

if (pybind11_FOUND)
    find_package_message(pybind11 "Found pybind11: ${pybind11_DIR} ${pybind11_INCLUDE_DIR} (version ${pybind11_VERSION})" "[${pybind11_DIR}][${pybind11_INCLUDE_DIR}]")
endif()

# when the user specifies CMAKE_INSTALL_PREFIX on first configure, install to "hoomd" under that prefix
set(PYTHON_SITE_INSTALL_DIR "hoomd" CACHE PATH
    "Python site-packages directory (relative to CMAKE_INSTALL_PREFIX)")

# when no CMAKE_INSTALL_PREFIX is set, default to python's install prefix
if((CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR "${_python_prefix}" STREQUAL "${CMAKE_INSTALL_PREFIX}") AND Python_SITEARCH)
    string(LENGTH "${_python_prefix}" __python_prefix_len)
    string(SUBSTRING "${Python_SITEARCH}" 0 ${__python_prefix_len} _python_site_package_prefix)
    math(EXPR _shart_char "${__python_prefix_len}+1")
    string(SUBSTRING "${Python_SITEARCH}" ${_shart_char} -1 _python_site_package_rel)
    string(COMPARE EQUAL "${_python_site_package_prefix}" "${_python_prefix}" _prefix_equal)
    if (NOT _prefix_equal)
        message(STATUS "Python site packages (${Python_SITEARCH}) does not start with python prefix (${_python_prefix})")
        message(STATUS "HOOMD may not install to the correct location")
    endif()

    set(CMAKE_INSTALL_PREFIX "${_python_prefix}" CACHE PATH "HOOMD installation path" FORCE)
    set(PYTHON_SITE_INSTALL_DIR "${_python_site_package_rel}/hoomd" CACHE PATH
        "Python site-packages directory (relative to CMAKE_INSTALL_PREFIX)" FORCE)
endif()

find_package_message(hoomd_install "Installing hoomd python module to: ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}" "[${CMAKE_INSTALL_PREFIX}][${PYTHON_SITE_INSTALL_DIR}]")
