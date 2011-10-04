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

/*! \file MolFilePlugin.cc
    \brief Defines MolFilePlugin class and contains molfile plugin specific
    methods of the FileFormatManager class.
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "MolFilePlugin.h"
#include "FileFormatManager.h"

#include "vmdplugin.h"
#include "molfile_plugin.h"

// platform neutral interface to loading shared objects.
#include "vmddlopen.h"

#include <iostream>
using namespace std;

extern "C" 
    {
    //! define C binding for plugin initialize function pointer
    typedef int (*initfunc)(void);
    //! define C binding for plugin register function pointer
    typedef int (*regfunc)(void *, vmdplugin_register_cb);
    //! define C binding for plugin release function pointer
    typedef int (*finifunc)(void);
    }

/*! Plugin registration callback function
  \param v opaque pointer to the plugin manager class
  \param plugin pointer to the plugin struct to be registered.
  \return -1 if failed, 0 if successful or plugin already registerd.
*/
static int molfile_register_cb(void *v, vmdplugin_t *plugin) 
    {
    FileFormatManager *mgr = (FileFormatManager *)v;

    // check that the ABI version matches
    if (plugin->abiversion != vmdplugin_ABIVERSION) 
        {
        cerr << "Rejecting plugin with incorrect ABI version: expected "
                  << vmdplugin_ABIVERSION << " and found " << plugin->abiversion
                  << endl;
        return -1;
        }

    // check that there are no null terms in the plugin
    if (!plugin->type || !plugin->name || !plugin->author) 
        {
        cerr << "Rejecting plugin with required header values set to NULL." << endl;
        return -1;
        }

    // check new plugin against already-loaded plugins.
    // return value -1 means "don't load", 0 means "add", 
    // >0 means "replace" plugin at index idx.
    int idx = mgr->check_molfile_plugin(plugin);
    if (idx < 0)
        return 0;

    mgr->add_molfile_plugin(idx, new MolFilePlugin(plugin));
    return 0;
    }

/*! Public MolFilePlugin constructor. 
  \param p opaque pointer to the plugin struct.

  \note To be called from the plugin register callback.
*/
MolFilePlugin::MolFilePlugin(void *p)
    {
    if (p != 0) 
        {
        molfile_plugin_t *mfp = (molfile_plugin_t *)p;
        m_major = mfp->majorv;
        m_minor = mfp->minorv;
        format_type = string(mfp->name);
        format_name = string(mfp->prettyname);
        plugin = p;
        handle = NULL;
        }
    else
        {
        plugin = 0;
        m_major = 0;
        m_minor = -1;
        format_type = string("dummy");
        format_name = string("Dummy molfile plugin");
        handle = NULL;
        }
    }

MolFilePlugin::MolFilePlugin(const MolFilePlugin &p)
    {
    plugin = p.plugin;
    handle = NULL;
    }

MolFilePlugin::~MolFilePlugin()
    {
    // close open file and release references.
    }

////////////////////////////////////////////////////////////////////////////////
// VMD molfile specific methods from the FileFormatManager class.
// better to put them here, so that we don't clutter too many 
// files with molfile specific codes and includes.
////////////////////////////////////////////////////////////////////////////////

/*! Test whether a plugin is suitable. 
  \param p opaque pointer to the plugin struct.
  \return 0 if the plugin should be appended to the list of plugins,
  -1 if the same or a newer plugin is already available, and
  any other number >0 if the plugin with that index should be replaces.

  We first check if a plugin with the same type string already 
  exists and in that case the plugin has to have a higher version
  number to be accepted.
*/
int FileFormatManager::check_molfile_plugin(void *p) const
    {
    molfile_plugin_t *mfp = (molfile_plugin_t *)p;

    // check if a plugin of the same file type already exists and return
    // its index number in case 
    for (unsigned int i=0; i < m_proxy_list.size(); ++i)
        {
        if (m_proxy_list[i]->get_type() == string(mfp->name))
            {
            if (m_proxy_list[i]->check_version(mfp->majorv, mfp->minorv) > 0)
                return i;
            else
                return -1;
            }
        }
    return 0;
    }

/*!
  \param dsofile filename of the shared object to load
*/
void FileFormatManager::loadMolFileDSOFile(std::string dsofile)
    {
    void *handle = vmddlopen(dsofile.c_str());
    if (!handle)
        {
        cerr << "Unable to open dynamic library file '" << dsofile << "': "
             << vmddlerror() << endl;
        return;
        }
    
    // need to check if we already have this same handle open.
    // we would get a segfault when loading the same symbols again.
    for (unsigned int i=0; i < m_dsohandle_list.size(); ++i)
        {
        if (m_dsohandle_list[i] == handle)
            {
            cout << "Already have a handle to the shared library " 
                 << dsofile << ". No further action required." << endl;
            return;
            }
        }
    
    // load and execute initialization function.
    void *ifunc = vmddlsym(handle, "vmdplugin_init");
    if (!ifunc)
        {
        cerr << "symbol vmdplugin_init not found in " << dsofile << ". "
             << "no plugin(s) loaded." << endl;
        return;
        }
    
    if (((initfunc)(ifunc))())
        {
        cerr << "vmdplugin_init() for " << dsofile << " failed. "
             << "no plugin(s) loaded." << endl;
        vmddlclose(handle);
        return;
        }

    m_dsohandle_list.push_back(handle);
    m_dsotype_list.push_back(MOLFILE_PLUGIN);
    
    // register contained plugins.
    void *rfunc = vmddlsym(handle, "vmdplugin_register");
    if (!rfunc)
        {
        cerr << "symbol vmdplugin_register not found in " << dsofile << ". "
             << "no plugin(s) loaded." << endl;
        return;
        }
    else
        ((regfunc) rfunc)(this, molfile_register_cb);
    }

/*! Unload Molfile plugins DSO.
  \param handle handle to the dynamical shared object that contains the plugin(s).
*/
void FileFormatManager::unload_molfile_dso(void *handle)
    {
    void *ffunc = vmddlsym(handle, "vmdplugin_fini");
    if (ffunc) 
        ((finifunc)(ffunc))();
    vmddlclose(handle);
    }


#ifdef WIN32
#pragma warning( pop )
#endif

