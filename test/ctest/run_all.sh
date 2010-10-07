#! /bin/bash

# simple runner script to run all predefined ctest_hoomd scripts
ctest $* -S ctest_hoomd.cmake
ctest $* -S ctest_hoomd_double.cmake
ctest $* -S ctest_hoomd_single.cmake
ctest $* -S ctest_hoomd_static_single_cuda.cmake
