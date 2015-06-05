/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

string output_version_info()
    {
    ostringstream o;
    // output the version info that comes from CMake
    o << "HOOMD-blue " << HOOMD_VERSION_LONG;

    #ifdef ENABLE_CUDA
    int cudart_major = CUDART_VERSION / 1000;
    int cudart_minor = (CUDART_VERSION - cudart_major * 1000) / 10;

    o << " CUDA (" << cudart_major << "." << cudart_minor << ")";
    #endif

    #ifdef SINGLE_PRECISION
    o << " SINGLE";
    #else
    o << " DOUBLE";
    #endif

    #ifdef ENABLE_MPI
    o << " MPI";
    #endif

    #ifdef ENABLE_MPI_CUDA
    o << " MPI_CUDA";
    #endif

    #ifdef __SSE__
    o << " SSE";
    #endif

    #ifdef __SSE2__
    o << " SSE2";
    #endif

    #ifdef __SSE3__
    o << " SSE3";
    #endif

    #ifdef __SSE4_1__
    o << " SSE4_1";
    #endif

    #ifdef __SSE4_2__
    o << " SSE4_2";
    #endif

    #ifdef __AVX__
    o << " AVX";
    #endif

    #ifdef __AVX2__
    o << " AVX2";
    #endif

    o << endl;

    // output the compiled date and copyright information
    o << "Compiled: " << COMPILE_DATE << endl;
    o << "Copyright 2009-2015 The Regents of the University of Michigan." << endl << endl;

    // output the paper citation information
    o << "All publications and presentations based on HOOMD-blue, including any reports" << endl;
    o << "or published results obtained, in whole or in part, with HOOMD-blue, will" << endl;
    o << "acknowledge its use according to the terms posted at the time of submission on:" << endl;
    o << "http://codeblue.umich.edu/hoomd-blue/citations.html" << endl;

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
