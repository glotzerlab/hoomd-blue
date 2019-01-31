// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "HOOMDVersion.h"
#include <iostream>
#include <sstream>
#include <string>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;

/*! \file HOOMDVersion.cc
    \brief Defines functions for formatting compile time version information as a string.

    \ingroup utils
*/

std::string hoomd_compile_flags()
    {
    ostringstream o;

    #ifdef ENABLE_CUDA
    int cudart_major = CUDART_VERSION / 1000;
    int cudart_minor = (CUDART_VERSION - cudart_major * 1000) / 10;

    o << "CUDA (" << cudart_major << "." << cudart_minor << ") ";
    #endif

    #ifdef SINGLE_PRECISION
    o << "SINGLE ";
    #else
    o << "DOUBLE ";
    #ifdef ENABLE_HPMC_MIXED_PRECISION
    o << "HPMC_MIXED ";
    #endif
    #endif

    #ifdef ENABLE_MPI
    o << "MPI ";
    #endif

    #ifdef ENABLE_MPI_CUDA
    o << "MPI_CUDA ";
    #endif

    #ifdef ENABLE_TBB
    o << "TBB ";
    #endif

    #ifdef __SSE__
    o << "SSE ";
    #endif

    #ifdef __SSE2__
    o << "SSE2 ";
    #endif

    #ifdef __SSE3__
    o << "SSE3 ";
    #endif

    #ifdef __SSE4_1__
    o << "SSE4_1 ";
    #endif

    #ifdef __SSE4_2__
    o << "SSE4_2 ";
    #endif

    #ifdef __AVX__
    o << "AVX ";
    #endif

    #ifdef __AVX2__
    o << "AVX2 ";
    #endif

    #ifdef ALWAYS_USE_MANAGED_MEMORY
    o << "ALWAYS_MANAGED ";
    #endif

    return o.str();
    }

string output_version_info()
    {
    ostringstream o;
    // output the version info that comes from CMake
    o << "HOOMD-blue " << HOOMD_VERSION_LONG;

    o << " " << hoomd_compile_flags();

    o << endl;

    // output the compiled date and copyright information
    o << "Compiled: " << COMPILE_DATE << endl;
    o << "Copyright (c) 2009-2019 The Regents of the University of Michigan." << endl;

    // warn the user if they are running a debug or GPU emulation build
#ifndef NDEBUG
    o << endl << "WARNING: This is a DEBUG build, expect slow performance." << endl;
#endif

#ifdef ENABLE_CUDA
#ifdef _DEVICEEMU
    o << endl << "WARNING: This is a GPU emulation build, expect extremely slow performance." << endl;
#endif
#endif
    return o.str();
    }
