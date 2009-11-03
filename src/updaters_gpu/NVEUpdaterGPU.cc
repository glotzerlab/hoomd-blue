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

/*! \file NVEUpdaterGPU.cc
    \brief Defines the NVEUpdaterGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "NVEUpdaterGPU.h"
#include "NVEUpdaterGPU.cuh"

#include <boost/bind.hpp>
using namespace boost;

#include <boost/python.hpp>
using namespace boost::python;

using namespace std;

/*! \param sysdef System to update
    \param deltaT Time step to use
*/
NVEUpdaterGPU::NVEUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT) : NVEUpdater(sysdef, deltaT)
    {
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    // at least one GPU is needed
    if (exec_conf.gpu.size() == 0)
        {
        cerr << endl << "***Error! Creating a NVEUpdaterGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing NVEUpdaterGPU");
        }
    }

/*! \param timestep Current time step of the simulation

    Calls gpu_nve_pre_step and gpu_nve_step to do the dirty work.
*/
void NVEUpdaterGPU::update(unsigned int timestep)
    {
    assert(m_pdata);
    
    // if we haven't been called before, then the accelerations have not been set and we need to calculate them
    if (!m_accel_set)
        {
        m_accel_set = true;
        // calculate the accelerations of the particles at the initial time step
        computeAccelerations(timestep, "NVE");
        }
        
    if (m_prof)
        m_prof->push(exec_conf, "NVE");
        
    // access the particle data arrays
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    
    if (m_prof) m_prof->push(exec_conf, "Half-step 1");
    
    // call the pre-step kernel on all GPUs in parallel
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(gpu_nve_pre_step, d_pdata[cur_gpu], box, m_deltaT, m_limit, m_limit_val));
        
    exec_conf.syncAll();
    
    uint64_t mem_transfer = m_pdata->getN() * (16+32+16+48);
    uint64_t flops = m_pdata->getN() * (15+3+9+15);
    if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
    
    // release the particle data arrays so that they can be accessed to add up the accelerations
    m_pdata->release();
    
    // communicate the updated positions among the GPUs
    m_pdata->communicatePosition();
    
    // functions that computeAccelerations calls profile themselves, so suspend
    // the profiling for now
    if (m_prof) m_prof->pop(exec_conf);
    
    // for the next half of the step, we need the net force at t+deltaT
    computeNetForceGPU(timestep+1, "NVE");
    
    if (m_prof) m_prof->push(exec_conf, "NVE");
    if (m_prof) m_prof->push(exec_conf, "Half-step 2");
    
    // get the particle data arrays again so we can update the 2nd half of the step
    d_pdata = m_pdata->acquireReadWriteGPU();
    // also access the net force data for reading
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
    
    // call the post-step kernel on all GPUs in parallel
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(gpu_nve_step, d_pdata[cur_gpu], d_net_force.data, m_deltaT, m_limit, m_limit_val));
        
    exec_conf.syncAll();
    m_pdata->release();
    
    // and now the acceleration at timestep+1 is precalculated for the first half of the next step
    if (m_prof)
        {
        mem_transfer = m_pdata->getN() * (16 + 4 + 16 + 32);
        flops = m_pdata->getN() * (3 + 6);
        m_prof->pop(exec_conf, flops, mem_transfer);
        m_prof->pop();
        }
    }

void export_NVEUpdaterGPU()
    {
    class_<NVEUpdaterGPU, boost::shared_ptr<NVEUpdaterGPU>, bases<NVEUpdater>, boost::noncopyable>
    ("NVEUpdaterGPU", init< boost::shared_ptr<SystemDefinition>, Scalar >())
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

