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
// Maintainer: akohlmey

/*! \file FileFormatProxy.h
    \brief Declares the FileFormatProxy base class for 
    transparent i/o to molecular file formats.
*/

#ifndef __FILE_FORMAT_PROXY_H__
#define __FILE_FORMAT_PROXY_H__

#include <string>

/*! Proxy class for molecule file i/o, either through through compiled
  classes, external libraries, or the molfile plugins from VMD.
  Each instance of FileFormatProxy acts as a proxy for the corresponding
  API and thus provides a consistent and independent interface to the 
  required functionality, as far it is supported.
  
  \ingroup utils
*/
class FileFormatProxy
    {
    public:
        //! Constructor
        FileFormatProxy();
        //! Destructor
        virtual ~FileFormatProxy();

    enum { FCAP_NONE=0, 
           FCAP_ATOMTYPE=1<<0,
           FCAP_POSITION=1<<1, 
           FCAP_VELOCITY=1<<2, 
           FCAP_CELL=1<<3,
           FCAP_MASS=1<<4,
           FCAP_CHARGE=1<<4,
           FCAP_BONDDEF=1<<5,
           FCAP_BONDTYPE=1<<6,
           FCAP_ANGLEDEF=1<<7,
           FCAP_ANGLETYPE=1<<8,
           FCAP_DIHEDRALDEF=1<<8,
           FCAP_DIHEDRALTYPE=1<<9,
           FCAP_IMPROPERDEF=1<<10,
           FCAP_IMPROPERTYPE=1<<11,
           FCAP_APPEND=1<<30,
           FCAP_WALL=1<<31 };

    protected:
        //! Copy Constructor
        FileFormatProxy(const FileFormatProxy& proxy);
        
    public: 
        //! Returns the canonical file format name.
        std::string get_type() const { return format_type; }
        
        //! Returns a nicely formatted name for the file format.
        std::string get_name() const { return format_name; }

        //! Compares format version information against external numbers.
        int check_version(const int major, const int minor) const;
        
    protected:
        std::string format_type;   //!< canonical type name of the file format
        std::string format_name;   //!< "pretty" name of the file format
        std::string file_name;     //!< name of file managed by this class
        void *handle;              //!< opaque data handle of an i/o stream
        int  m_major;              //!< major version number of the format module
        int  m_minor;              //!< minor version number of the format module
        unsigned int m_read_caps;  //!< reading capability bitmask of this format
        unsigned int m_write_caps; //!<< writing capability bitmast of this format
    };

#endif

