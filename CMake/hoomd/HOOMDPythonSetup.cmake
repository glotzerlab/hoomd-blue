## Sets up python for hoomd

# macro for running python and getting output
macro(run_python code result)
execute_process(
    COMMAND
    ${PYTHON_EXECUTABLE} -c ${code}
    OUTPUT_VARIABLE ${result}
    RESULT_VARIABLE PY_ERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

if(PY_ERR)
    message(STATUS "Error while querying python for information")
endif(PY_ERR)
endmacro(run_python)

# find the python interpreter, first
find_package(PythonInterp REQUIRED)

# get the python installation prefix and version
run_python("import sys\; print('%d.%d' % (sys.version_info[0],sys.version_info[1]))" _python_version)
string(REPLACE "." "" _python_version_no_dots ${_python_version})

# determine the include directory
if (_python_version VERSION_GREATER 3)
    run_python("import sysconfig\; print(sysconfig.get_path('include'))" _python_include_hint)
    run_python("import sysconfig\; print(sysconfig.get_config_var('LIBDIR'))" _python_lib_hint)
    run_python("import sysconfig\; print(sysconfig.get_config_var('LDLIBRARY'))" _python_dynamic_lib_name)
else()
    run_python("from distutils import sysconfig\; print sysconfig.get_python_inc()" _python_include_hint)
    run_python("from distutils import sysconfig\; print sysconfig.PREFIX" _python_prefix_hint)
    run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LIBPL')" _python_static_hint)
    run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LIBRARY')" _python_static_lib_name)
    run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LDLIBRARY')" _python_dynamic_lib_name)
endif()

# always link the dynamic python library
get_filename_component(_python_lib_first ${_python_dynamic_lib_name} NAME)

find_path(PYTHON_INCLUDE_DIR Python.h
          HINTS ${_python_include_hint}
          NO_DEFAULT_PATH)

# add a blank suffix to the beginning to find the Python framework
set(_old_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ";${CMAKE_FIND_LIBRARY_SUFFIXES}")
find_library(PYTHON_LIBRARY
             NAMES ${_python_lib_first} python${_python_version_no_dots} python${_python_version}
             HINTS ${_python_prefix_hint} ${_python_static_hint} ${_python_lib_hint}
             PATH_SUFFIXES lib64 lib libs
             NO_DEFAULT_PATH
             )
set(${CMAKE_FIND_LIBRARY_SUFFIXES} _old_suffixes)

MARK_AS_ADVANCED(
  PYTHON_LIBRARY
  PYTHON_INCLUDE_DIR
  BOOST_PYTHON_COMPONENT
)

SET(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
SET(PYTHON_LIBRARIES "${PYTHON_LIBRARY}")

if (_python_version VERSION_GREATER 3)
    SET(BOOST_PYTHON_COMPONENT "python3")
else()
    SET(BOOST_PYTHON_COMPONENT "python")
endif()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PythonLibs DEFAULT_MSG PYTHON_LIBRARIES PYTHON_INCLUDE_DIRS)
