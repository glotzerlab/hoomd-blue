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
    // at least one GPU is needed
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a Enforce2DUpdaterGPU with no GPU in the execution configuration" << endl;
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
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
    
    // call the enforce 2d kernel
    gpu_enforce2d(m_pdata->getN(),
                  d_vel.data,
                  d_accel.data);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(exec_conf);
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

