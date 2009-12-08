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

// $Id: Enforce2DUpdaterGPU.cc 2148 2009-10-07 20:05:29Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/src/updaters_gpu/Enforce2DUpdaterGPU.cc $
// Maintainer: joaander

/*! \file Enforce2DUpdaterGPU.cc
    \brief Defines the Enforce2DUpdaterGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "Enforce2DUpdaterGPU.h"
#include "Enforce2DUpdaterGPU.cuh"

#include <boost/bind.hpp>
using namespace boost;

#include <boost/python.hpp>
using namespace boost::python;

using namespace std;

/*! \param sysdef System to update
*/
Enforce2DUpdaterGPU::Enforce2DUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef) : Enforce2DUpdater(sysdef)
    {
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    // at least one GPU is needed
    if (exec_conf.gpu.size() == 0)
        {
        cerr << endl << "***Error! Creating a Enforce2DUpdaterGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing Enforce2DUpdaterGPU");
        }
    }

/*! \param timestep Current time step of the simulation

    Calls gpu_enforce2d to do the actual work.
*/
void Enforce2DUpdaterGPU::update(unsigned int timestep)
    {
    assert(m_pdata);
            
    if (m_prof)
        m_prof->push(exec_conf, "Enforce2D");
        
    // access the particle data arrays
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    
    // call the enforce 2d kernel on all GPUs in parallel
    exec_conf.tagAll(__FILE__, __LINE__);
    
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(gpu_enforce2d, d_pdata[cur_gpu]));
    
    exec_conf.syncAll();
                        
    m_pdata->release();
    
    if (m_prof)
        m_prof->pop(exec_conf, "Enforce2D");
    }

void export_Enforce2DUpdaterGPU()
    {
    class_<Enforce2DUpdaterGPU, boost::shared_ptr<Enforce2DUpdaterGPU>, bases<Enforce2DUpdater>, boost::noncopyable>
    ("Enforce2DUpdaterGPU", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

