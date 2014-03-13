#! /bin/bash

# simple runner script to run all predefined ctest_hoomd scripts
ctest $* -S ctest-single-cpu.cmake
ctest $* -S ctest-double-cpu.cmake
ctest $* -S ctest-single-cpu-mpi.cmake
ctest $* -S ctest-double-cpu-mpi.cmake

ctest $* -S ctest-single-cuda.cmake
ctest $* -S ctest-double-cuda.cmake
ctest $* -S ctest-single-cuda-mpi.cmake
ctest $* -S ctest-double-cuda-mpi.cmake
