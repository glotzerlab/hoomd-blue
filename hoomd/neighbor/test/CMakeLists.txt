# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

###################################
## Setup all of the test executables
#
# Example for foo_bar_test.cc:
#   foo_bar
set(TEST_LIST
    lbvh
    )
###################################

macro(compile_test TEST_EXE TEST_SRC)
    # add and link the unit test executable
    add_executable(${TEST_EXE} EXCLUDE_FROM_ALL ${TEST_SRC})
    add_dependencies(test_all ${TEST_EXE})
    target_link_libraries(${TEST_EXE} ${COMPONENT_NAME} ${HOOMD_LIBRARIES} ${PYTHON_LIBRARIES})
    fix_cudart_rpath(${TEST_EXE})

    if(ENABLE_MPI)
        # set appropriate compiler/linker flags
        if(MPI_CXX_COMPILE_FLAGS)
            set_target_properties(${TEST_EXE} PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
        endif(MPI_CXX_COMPILE_FLAGS)
        if(MPI_CXX_LINK_FLAGS)
            set_target_properties(${TEST_EXE} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
        endif(MPI_CXX_LINK_FLAGS)
    endif(ENABLE_MPI)
endmacro(compile_test)

# add tests
foreach(CUR_TEST ${TEST_LIST})
    set(CUR_TEST_SRC ${CUR_TEST}_test.cc)
    set(CUR_TEST_EXE ${CUR_TEST}_test)
    set(CUR_TEST_NAME neighbor-core-${CUR_TEST})

    compile_test(${CUR_TEST_EXE} ${CUR_TEST_SRC})
    if(ENABLE_MPI)
        add_test(NAME ${CUR_TEST_NAME} COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_POSTFLAGS} $<TARGET_FILE:${CUR_TEST_EXE}>)
    else()
        add_test(NAME ${CUR_TEST_NAME} COMMAND $<TARGET_FILE:${CUR_TEST_EXE}>)
    endif()
endforeach(CUR_TEST)
