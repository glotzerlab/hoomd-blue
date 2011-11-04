/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

* All publications based on HOOMD-blue, including any reports or published
results obtained, in whole or in part, with HOOMD-blue, will acknowledge its use
according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website
at: http://codeblue.umich.edu/hoomd-blue/.

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


// Maintainer: akohlmey

/*! \file FileFormatProxy.cc
    \brief Implements the FileFormatProxy class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "FileFormatProxy.h"

using namespace std;

/*! Public FileFormatProxy constructor. 
 */
FileFormatProxy::FileFormatProxy()
    : handle(0), m_major(0), m_minor(-1), m_read_caps(FCAP_NONE), m_write_caps(FCAP_NONE)
    {
    format_type = string("dummy");
    format_name = string("Dummy File Format");
    }

/*! Copy constructor
  \param p proxy class to be copied.
  
  The general strategy is the following: Each FileFormatProxy class (or one derived from it) is responsible 
  for just one file i/o stream.  When a new i/o request is made, a copy of the class is generated.  The 
  FileFormatManager class manages the master list and is usually queried to create a copy of theproxy,
 
  \note open files/handles must not be copied.
*/
FileFormatProxy::FileFormatProxy(const FileFormatProxy& p) 
    : format_type(p.format_type), format_name(p.format_name), handle(0), m_major(p.m_major), m_minor(p.m_minor),
      m_read_caps(p.m_read_caps), m_write_caps(p.m_write_caps)
    {
    }

FileFormatProxy::~FileFormatProxy()
    {
    // XXX: close active file and release references if needed.
    }

/*! Compares format version information against external numbers.
  \param major major version number of external plugin
  \param minor minor version number of external plugin
  \return  1, 0, or -1, if the new plugin version is higher, same, or lower.
  \note if m_major of a file format is negative, this format must not be overridden.
*/
int FileFormatProxy::check_version(const int major, const int minor) const 
            {
            // immutable file format.
            if (m_major < 0)
                return -1;
            
            if (major > m_major)
                return 1;
            else if (major <= m_major)
                return -1;
            else if (minor > m_minor) 
                return 1;
            else if (minor < m_minor)
                return -1;
            else 
                return 0;
            }

#ifdef WIN32
#pragma warning( pop )
#endif

