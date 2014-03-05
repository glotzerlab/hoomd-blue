/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

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

// we need to include boost.python in order to export ExampleUpdater to python
#include <boost/python.hpp>
using namespace boost::python;

// we need to include boost.bind for GPUWorker execution
#include <boost/bind.hpp>
using namespace boost;

#include "ExampleUpdater.h"
#ifdef ENABLE_CUDA
#include "ExampleUpdater.cuh"
#endif

/*! \file ExampleUpdater.cc
    \brief Definition of ExampleUpdater
*/

// ********************************
// here follows the code for ExampleUpdater on the CPU

/*! \param sysdef System to zero the velocities of
*/
ExampleUpdater::ExampleUpdater(boost::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef)
    {
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void ExampleUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ExampleUpdater");
    
    // access the particle data for writing on the CPU
    assert(m_pdata);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    
    // zero the velocity of every particle
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        h_vel.data[i].x = Scalar(0.0);
        h_vel.data[i].y = Scalar(0.0);
        h_vel.data[i].z = Scalar(0.0);
        }
    
    if (m_prof) m_prof->pop();
    }

void export_ExampleUpdater()
    {
    class_<ExampleUpdater, boost::shared_ptr<ExampleUpdater>, bases<Updater>, boost::noncopyable>
    ("ExampleUpdater", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

// ********************************
// here follows the code for ExampleUpdater on the GPU

#ifdef ENABLE_CUDA

/*! \param sysdef System to zero the velocities of
*/
ExampleUpdaterGPU::ExampleUpdaterGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : ExampleUpdater(sysdef)
    {
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void ExampleUpdaterGPU::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ExampleUpdater");
    
    // access the particle data arrays for writing on the GPU
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    
    // call the kernel devined in ExampleUpdater.cu
    gpu_zero_velocities(d_vel.data, m_pdata->getN());
    
    // check for error codes from the GPU if error checking is enabled
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    if (m_prof) m_prof->pop();
    }

void export_ExampleUpdaterGPU()
    {
    class_<ExampleUpdaterGPU, boost::shared_ptr<ExampleUpdaterGPU>, bases<ExampleUpdater>, boost::noncopyable>
    ("ExampleUpdaterGPU", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }

#endif // ENABLE_CUDA
