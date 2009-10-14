// -*- c++ -*-
// Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
// (HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
// Iowa State University and The Regents of the University of Michigan All rights
// reserved.
// 
// HOOMD-blue may contain modifications ("Contributions") provided, and to which
// copyright is held, by various Contributors who have granted The Regents of the
// University of Michigan the right to modify and/or distribute such Contributions.
// 
// Redistribution and use of HOOMD-blue, in source and binary forms, with or
// without modification, are permitted, provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright notice, this
// list of conditions, and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions, and the following disclaimer in the documentation and/or
// other materials provided with the distribution.
// 
// * Neither the name of the copyright holder nor the names of HOOMD-blue's
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
// 
// Disclaimer
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
// ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.
// 
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 

// $Id$
// $URL$
// Maintainer: akohlmey

/*! \file FileFormatManager.h
	\brief Declares the MolFilePlugin and FileFormatManager classes
*/

#ifndef __FILE_FORMAT_MANAGER_H__
#define __FILE_FORMAT_MANAGER_H__

#include "FileFormatProxy.h"
#include "MolFilePlugin.h"

#include <string>
#include <vector>


//! Class to manage I/O to various file formats.
/*! 
  \ingroup utils
*/
class FileFormatManager
    {
    public:
        //! Constructor
        FileFormatManager();
        //! Virtual destructor
        ~FileFormatManager();

    public: // n.b. the following methods have to be public so they can be called
            // from the molfile register callback function. 
        //! Check whether a new molfile plugin should be added to the list.
        int check_molfile_plugin(void *p) const;

        //! Add a new molfile plugin to the list of plugins at index idx.
        //  idx = 0 means append, idx < 0 mean ignore.
        int add_molfile_plugin(const unsigned int idx, MolFilePlugin *p);

    private:
        //! list of format types supported by the file format manager.
        enum { NONE=0, MOLFILE_PLUGIN, NATIVE_XML, NATIVE_DCD, NATIVE_MOL2 };

        //! unload shared objects for molfile plugins
        void unload_molfile_dso(void *handle);

    public:
        //! Load molfile plugin(s) from shared object of the given name.
        void loadMolFileDSOFile(std::string dsofile);

        //! fide a matching plugin by giving the plugin type. 
        // If the type is NULL, try to guess from file name extension.
        FileFormatProxy *findFormat(const char *type, const char *filename);
        
    private:
        std::vector<FileFormatProxy *> m_proxy_list; //!< list of available I/O format classes.
        std::vector<int> m_ftype_list;               //!< list of format types corresponding to proxy classes.
        std::vector<void *> m_dsohandle_list;        //!< list of DSO handles for plugins.
        std::vector<int> m_dsotype_list;             //!< list of types corresponding to DSO handles.
    };

#endif
