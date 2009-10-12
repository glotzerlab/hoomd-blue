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

/*! \file MolFilePluginMgr.h
	\brief Declares the MolFilePlugin and MolFilePluginMgr classes
*/

#ifndef __MOLFILE_PLUGIN_MGR_H__
#define __MOLFILE_PLUGIN_MGR_H__

#include <string>
#include <vector>

/*! Interface class for file format i/o through the molfile plugins from VMD.
  Each instance of MolFilePlugin acts as a proxy for the corresponding
  plugin from the molfile library and provides a plugin API independent
  interface to the required functionality, as far it is supported.
  
  \ingroup utils
*/
class MolFilePlugin
	{
    public:
        //! Constructor
        MolFilePlugin(void *p);
        //! Destructor
        virtual ~MolFilePlugin();

    protected:
        //! Copy Constructor (only to be accessed from MolFilePluginMgr)
        MolFilePlugin(const MolFilePlugin& plugin);
    
    private:
        //! Default constructor is disabled
        MolFilePlugin();

    public: 
        //! Returns the file format string of a plugin
        std::string get_type() const 
            { 
            return plugin_type;
            }
        
        //! Returns a nicely formatted name for the plugin
        std::string get_name() const
            {
            return plugin_name;
            }
        
        /*! Compares plugin version information against external numbers.
          \param major major version number of external plugin
          \param minor minor version number of external plugin
          \return  1, 0, or -1, if the new plugin version is higher, same, or lower.
        */
        int check_version(const int major, const int minor) const 
            {
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
        
    private:
        std::string plugin_type;   //!< canonical type name of the plugin
        std::string plugin_name;   //!< "pretty" name of the plugin.
        void *handle;              //!< data handle of the plugin.
        void *plugin;              //!< pointer to plugin.
        int  m_major;              //!< major version number of the plugin.
        int  m_minor;              //!< minor version number of the plugin.
    };


//! Class to manage molfile plugins from VMD.
/*! 
  \ingroup utils
*/
class MolFilePluginMgr
    {
    public:
        //! Constructor
        MolFilePluginMgr();
        //! Virtual destructor
        ~MolFilePluginMgr();

        //! Check whether a new plugin should be added to the list.
        int check_plugin(void *p) const;

        //! Add a new plugin to the list of plugins at index idx.
        //  idx = 0 means append, idx < 0 mean ignore.
        int add_plugin(const unsigned int idx, MolFilePlugin *p);

        //! Load plugin(s) from shared object of the given name.
        void loadDSOFile(std::string dsofile);

        //! file a matching plugin by giving the plugin type. 
        // If the type is NULL, try guess from file name.
        MolFilePlugin *findPlugin(const char *type, const char *filename);
        
    private:
        std::vector<MolFilePlugin *> m_plist;   //!< list of available plugins.
        std::vector<void *> m_hlist;            //!< list of DSO handles;
    };

#endif
