/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

#include <iostream>
#include <string>

using namespace std;

#include "HOOMDVersion.h"
/*! \file HOOMDVersion.cc
	\brief Defines functions for writing compile time version information to the screen.
	
	\ingroup utils
*/

void output_version_info(bool verbose)
	{
	// output the version info differently if this is tagged as a subversion build or not
	if (HOOMD_SUBVERSION_BUILD)
		cout << "HOOMD svnversion " << HOOMD_SVNVERSION << endl;
	else
		cout << "HOOMD " << HOOMD_VERSION << endl;
	
	// output the compiled date and copyright information
	cout << "Compiled: " << COMPILE_DATE << endl;
	cout << "Copyright, 2008, Ames Laboratory Iowa State University" << endl;
	
	// output the paper citation information
	cout << "-----" << endl;
	cout << "http://www.ameslab.gov/hoomd/" << endl;
	cout << "This code is the implementation of the algorithms discussed in:" << endl;
	cout << "   Joshua A. Anderson, Chris D. Lorenz, and Alex Travesset - 'General" << endl;
	cout << "   Purpose Molecular Dynamics Fully Implemented on Graphics Processing" << endl;
	cout << "   Units', to appear in the Journal of Computational Physics" << endl;
	cout << "-----" << endl;

	// warn the user if they are running a debug or GPU emulation build
	#ifndef NDEBUG
	cout << "WARNING: This is a DEBUG build, expect slow performance." << endl;
	#endif
	
	#ifdef USE_CUDA
	if (string(CUDA_BUILD_TYPE) != string("Device"))
		cout << "WARNING: This is a GPU emulation build, expect extremely slow performance." << endl;
	#endif
	
	#ifdef USE_CUDA_BUG_WORKAROUND
	cout << "WARNING: CUDA Bug workaround is in place. GPU performance is 1/2 of what it should be." << endl;
	#endif

	// if verbose output was requested, let the user know how everything was configured for the build
	if (verbose)
		{
		// set all the options to a default and modify them within ifdefs
		string use_cuda = "no";
		string use_python = "no";
		string use_sse = "no";
		string precision = "double";
		string use_static = "no";

		#ifdef USE_CUDA
		use_cuda = "yes";
		#endif

		#ifdef USE_PYTHON
		use_python = "yes";
		#endif

		#ifdef USE_SSE
		use_sse = "yes";
		#endif

		#ifdef SINGLE_PRECISION
		precision = "single";
		#endif

		#ifdef USE_STATIC
		use_static = "yes";
		#endif

		cout << "CUDA enabled:             " << use_cuda << endl;
		cout << "python module enabled:    " << use_python << endl;
		cout << "SSE instructions enabled: " << use_sse << endl;
		cout << "Floating point precision: " << precision << endl;
		cout << "Static library build:     " << use_static << endl;

		cout << endl;
		}
	}
