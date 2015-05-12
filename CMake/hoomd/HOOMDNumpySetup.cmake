# sets the variable NUMPY_INCLUDE_DIR for using numpy with python

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

if(NOT NUMPY_INCLUDE_DIR)

if (PYTHON_VERSION VERSION_GREATER 3)
    run_python("import numpy\; print(numpy.get_include())" NUMPY_INCLUDE_GUESS)
else()
    run_python("import numpy\; print numpy.get_include()" NUMPY_INCLUDE_GUESS)
endif()

# We use the full path name (including numpy on the end), but
# Double-check that all is well with that choice.
find_path(
    NUMPY_INCLUDE_DIR
    numpy/arrayobject.h
    HINTS
    ${NUMPY_INCLUDE_GUESS}
    )

if (NUMPY_INCLUDE_DIR)
message(STATUS "Found numpy: ${NUMPY_INCLUDE_DIR}")
endif (NUMPY_INCLUDE_DIR)

endif(NOT NUMPY_INCLUDE_DIR)

if (NUMPY_INCLUDE_DIR)
mark_as_advanced(NUMPY_INCLUDE_DIR)
endif (NUMPY_INCLUDE_DIR)

include_directories(${NUMPY_INCLUDE_DIR})
add_definitions(-DPY_ARRAY_UNIQUE_SYMBOL=PyArrayHandle)
add_definitions(-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)
