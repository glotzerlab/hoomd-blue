/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// $Id$
// $URL$
// Maintainer: joaander

/*! \file PathUtils.h
    \brief Simple functions for dealing with paths
*/

#ifndef __PATH_UTILS_H__
#define __PATH_UTILS_H__

#include <stdlib.h>
#include <string>
#include <stdexcept>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#ifdef UNIX
#include <unistd.h>
#endif

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

//! Get the path this executable is in
/*! \returns The path
    getExePath uses different methods on different platforms. It identifies the real file for the running executable.
*/
std::string getExePath()
    {
    // exe path identification from: http://stackoverflow.com/questions/1023306/finding-current-executables-path-without-proc-self-exe
    std::string result;

    #ifdef __APPLE__
    // ask darwin what our exe path is
    char buf[1024];
    memset(buf, 0, 1024);
    uint32_t bufsize = 1024;
    int retval = _NSGetExecutablePath(buf, &bufsize);
    if (retval != 0)
        throw std::runtime_error("Unable to determine executable path");

    // turn it into a real path
    char *realbuf = realpath(buf, NULL);
    result = std::string(realbuf);
    free(realbuf);
    
    #elif UNIX
    char buf[1024];
    memset(buf, 0, 1024);
    size_t bufsize = 1024;
    size_t retval = readlink("/proc/self/exe", buf, bufsize);
    if (retval == -1)
        throw std::runtime_error("Unable to determine executable path");

    result = std::string(buf);
    
    #elif WIN32
    #error Not implemented
    // see the above link for a howto on implementing this. Not a high priority to do so because windows is deprecated
    #endif

    // the above routines get the actual executable. Return the path to it
    #if (BOOST_VERSION <= 103500)
    return boost::filesystem::path(result).branch_path().native_file_string();
    #else
    return boost::filesystem::path(result).parent_path().native_file_string();
    #endif
    }

#endif

