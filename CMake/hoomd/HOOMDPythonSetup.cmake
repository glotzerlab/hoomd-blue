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

# determine the include directory
run_python("from distutils import sysconfig\; print sysconfig.get_python_inc()" _python_include_hint)
find_path(PYTHON_INCLUDE_DIR Python.h
          HINTS ${_python_include_hint}
          NO_DEFAULT_PATH)

# get the python installation prefix and version
run_python("from distutils import sysconfig\; print sysconfig.PREFIX" _python_prefix_hint)
run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LIBPL')" _python_static_hint)
run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LIBRARY')" _python_static_lib_name)
run_python("from distutils import sysconfig\; print sysconfig.get_config_var('LDLIBRARY')" _python_dynamic_lib_name)
run_python("from distutils import sysconfig\; print sysconfig.get_python_version()" _python_version)
string(REPLACE "." "" _python_version_no_dots ${_python_version})

if (ENABLE_STATIC)
    get_filename_component(_python_lib_first ${_python_static_lib_name} NAME)
else (ENABLE_STATIC)
    get_filename_component(_python_lib_first ${_python_dynamic_lib_name} NAME)
endif (ENABLE_STATIC)
message(STATUS "searching for first: " ${_python_lib_first})

# add a blank suffix to the beginning to find the Python framework
set(_old_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_SUFFIXES ";${CMAKE_FIND_LIBRARY_SUFFIXES}")
find_library(PYTHON_LIBRARY
             NAMES ${_python_lib_first} python${_python_version_no_dots} python${_python_version}
             HINTS ${_python_prefix_hint} ${_python_static_hint}
             PATH_SUFFIXES lib64 lib libs
             NO_DEFAULT_PATH
             )
set(${CMAKE_FIND_LIBRARY_SUFFIXES} _old_suffixes)

message(STATUS "python library " ${PYTHON_LIBRARY})

MARK_AS_ADVANCED(
  PYTHON_LIBRARY
  PYTHON_INCLUDE_DIR
)

SET(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
SET(PYTHON_LIBRARIES "${PYTHON_LIBRARY}")

