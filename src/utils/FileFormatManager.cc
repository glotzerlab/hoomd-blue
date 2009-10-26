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

/*! \file FileFormatManager.cc
    \brief Defines MolFilePlugin and FileFormatManager classes
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "FileFormatManager.h"

// platform neutral interface to loading shared objects.
#include "vmddlopen.h"

#include <iostream>
#include <boost/python.hpp>
using namespace boost::python;
using namespace std;


FileFormatManager::FileFormatManager()
    {
    // reserve some storage to reduce need for reallocation.
    m_proxy_list.reserve(64);
    m_ftype_list.reserve(64);
    m_dsohandle_list.reserve(64);
    m_dsotype_list.reserve(64);
    }

FileFormatManager::~FileFormatManager()
    {
    unsigned int i;
    
    // delete format proxy classes.
    for (i=0; i < m_proxy_list.size(); ++i)
        delete m_proxy_list[i];

    // release handles to DSOs.
    for (i=0; i < m_dsohandle_list.size(); ++i)
        {
        switch (m_dsotype_list[i])
            {
            case MOLFILE_PLUGIN:
                unload_molfile_dso(m_dsohandle_list[i]);
                break;

            case NONE: //fallthrough
            default:
                cerr << "Unknown DSO type. " << m_dsotype_list[i] << " Object not unloaded." << endl;
            }
        }
    }

/*!
 \param idx position at which the 
 \param p molfile plugin proxy class to be added
 \return index of plugin in the plugin list or -1.
*/
int FileFormatManager::add_molfile_plugin(const unsigned int idx, MolFilePlugin *p)
    {
    if (idx == 0)
        {
        int i=m_proxy_list.size();
        m_proxy_list.push_back(p);
        return i;
        }
    
    if (idx < m_proxy_list.size())
        {
        delete m_proxy_list[idx];
        m_proxy_list[idx] = p;
        return idx;
        }
    else
        {
        cerr << "Plugin Index " << idx << " out of range. Max.: " << m_proxy_list.size() -1 << endl;
        return -1;
        }
    }


//! exports the FileFormatManager class to python.
void export_FileFormatManager()
    {
    class_<FileFormatManager, boost::shared_ptr<FileFormatManager> >("FileFormatManager", init< >())
    .def("loadMolFileDSOFile", &FileFormatManager::loadMolFileDSOFile);
    }

#ifdef WIN32
#pragma warning( pop )
#endif
