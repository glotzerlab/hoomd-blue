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
// Maintainer: joaander

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
		cout << "HOOMD-blue svnversion " << HOOMD_SVNVERSION << endl;
	else
		cout << "HOOMD-blue " << HOOMD_VERSION << endl;
	
	// output the compiled date and copyright information
	cout << "Compiled: " << COMPILE_DATE << endl;
	cout << "Copyright 2008, 2009 Ames Laboratory Iowa State University and the Regents of the University of Michigan" << endl;
	
	// output the paper citation information
	cout << "-----" << endl;
	cout << "http://codeblue.umich.edu/hoomd-blue/" << endl;
	cout << "This code is the implementation of the algorithms discussed in:" << endl;
	cout << "   Joshua A. Anderson, Chris D. Lorenz, and Alex Travesset - 'General" << endl;
	cout << "   Purpose Molecular Dynamics Fully Implemented on Graphics Processing" << endl;
	cout << "   Units', Journal of Computational Physics 227 (2008) 5342-5359" << endl;
	cout << "-----" << endl;

	// warn the user if they are running a debug or GPU emulation build
	#ifndef NDEBUG
	cout << "WARNING: This is a DEBUG build, expect slow performance." << endl;
	#endif
	
	#ifdef ENABLE_CUDA
	#ifdef _DEVICEEMU
	cout << "WARNING: This is a GPU emulation build, expect extremely slow performance." << endl;
	#endif
	#endif
	}
