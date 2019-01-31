// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file global_array_test.cuh
    \brief Definitions of GPU kernel drivers for global_array_test.cc
    \ingroup unit_tests
*/

#ifndef __GLOBAL_ARRAY_TEST_CUH__
#define __GLOBAL_ARRAY_TEST_CUH__

extern "C"
    {

//! Adds one to every value in an array of ints
    cudaError_t gpu_add_one(int *d_data, unsigned int num);
//! Fills out the data array with a test pattern
    cudaError_t gpu_fill_test_pattern(int *d_data, unsigned int num);

    }

#endif
