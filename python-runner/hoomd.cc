/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
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

// Maintainer: joaander

// temporarily work around issues with the new boost fileystem libraries
// http://www.boost.org/doc/libs/1_46_1/libs/filesystem/v3/doc/index.htm

//! Enable old boost::filesystem API (temporary fix)
#define BOOST_FILESYSTEM_VERSION 2

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif


#include <boost/python.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>

using namespace boost::filesystem;

#include "PathUtils.h"
#include "HOOMDVersion.h"

#include <string>
#include <sstream>
#include <list>
#include <iostream>
using namespace std;

/*! \file hoomd.cc
    \brief Executable for running python scripts with the hoomd module builtin
*/

//! forward declaration of the inithoomd function that inits the python module from hoomd_module.cc
extern "C" void inithoomd();

//! A simple method for finding the hoomd_script python module
string find_hoomd_script()
    {
    // this works on the requirement in the hoomd build scripts that the python module is always
    // installed in lib/hoomd/python-module on linux and mac. On windows, it actually ends up in bin/python-module
    path exepath = path(getExePath());
    list<path> search_paths;
    search_paths.push_back(exepath / "python-module");                            // windows
    search_paths.push_back(exepath / ".." / "lib" / "hoomd" / "python-module");   // linux/mac
    search_paths.push_back(path(HOOMD_SOURCE_DIR) / "python-module");             // from source builds
    
    list<path>::iterator cur_path;
    for (cur_path = search_paths.begin(); cur_path != search_paths.end(); ++cur_path)
        {
        if (exists(*cur_path / "hoomd_script" / "__init__.py"))
            return cur_path->native_file_string();
        }
        
    cerr << endl 
         << "***Error! HOOMD python-module directory not found. Check your HOOMD directory structure." << endl;
    cerr << "Searched for hoomd_script in:" << endl;
    for (cur_path = search_paths.begin(); cur_path != search_paths.end(); ++cur_path)
        {
        cerr << cur_path->native_file_string() << endl;
        }
    
    return "";
    }

//! Main function for the executable
/*! \param argc argument count
    \param argv arguments
    Loads up the hoomd python module and then passes control onto Py_Main
*/
int main(int argc, char **argv)
    {
    if (argc == 1)
        {
        // This shell is an interactive launch with no arguments
        cout << "Launching intractive python shell now.... run \"from hoomd_script import *\" to load the HOOMD-blue python module" << endl << endl;
        }

    char module_name[] = "hoomd";
    PyImport_AppendInittab(module_name, &inithoomd);
    Py_Initialize();
    
    // Need to inject the hoomd module path and the plugins dir into sys.path
    string python_cmds("import sys\n");
    string hoomd_script_dir = find_hoomd_script();
    if (hoomd_script_dir != "")
        {
        python_cmds += string("sys.path.insert(0, r\"") + hoomd_script_dir + string("\")\n");
        }
    
    if (getenv("HOOMD_PLUGINS_DIR"))
        {
        string hoomd_plugins_dir = string(getenv("HOOMD_PLUGINS_DIR"));
        python_cmds += string("sys.path.insert(0, r\"") + hoomd_plugins_dir + string("\")\n");
        cout << "Notice: Using hoomd plugins in " << hoomd_plugins_dir << endl;
        }
        
    PyRun_SimpleString(python_cmds.c_str());
        
    int retval = Py_Main(argc, argv);
    
    // trying to clean up python's messy memory leaks
    Py_Finalize();
    return retval;
    }
#ifdef WIN32
#pragma warning( pop )
#endif

