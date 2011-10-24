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

#include <boost/python.hpp>
using namespace boost::python;
#include<boost/bind.hpp>
using namespace boost;

#include "TwoStepBerendsenGPU.h"
#include "TwoStepBerendsenGPU.cuh"

/*! \param sysdef System to which the Berendsen thermostat will be applied
    \param group Group of particles to which the Berendsen thermostat will be applied
    \param thermo Compute for themodynamic properties
    \param tau Time constant for Berendsen thermostat
    \param T Set temperature
*/
TwoStepBerendsenGPU::TwoStepBerendsenGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                         boost::shared_ptr<ParticleGroup> group,
                                         boost::shared_ptr<ComputeThermo> thermo,
                                         Scalar tau,
                                         boost::shared_ptr<Variant> T)
    : TwoStepBerendsen(sysdef, group, thermo, tau, T)
    {
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Creating a BerendsenGPU when CUDA is disabled" << endl << endl;
        throw std::runtime_error("Error initializing BerendsenGPU");
        }

    m_block_size = 256;
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TwoStepBerendsenGPU::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof)
        m_prof->push("Berendsen");

    // compute the current thermodynamic quantities and get the temperature
    m_thermo->compute(timestep);
    Scalar curr_T = m_thermo->getTemperature();

    // compute the value of lambda for the current timestep
    Scalar lambda = sqrt(Scalar(1.0) + m_deltaT / m_tau * (m_T->getValue(timestep) / curr_T - Scalar(1.0)));

    // access the particle data arrays for writing on the GPU
    gpu_pdata_arrays& d_pdata = m_pdata->acquireReadWriteGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the integration on the GPU
    gpu_berendsen_step_one(d_pdata,
                           d_index_array.data,
                           group_size,
                           box,
                           m_block_size,
                           lambda,
                           m_deltaT);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_pdata->release();

    if (m_prof)
        m_prof->pop();
    }

void TwoStepBerendsenGPU::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof)
        m_prof->push("Berendsen");

    // get the net force
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);

    // access the aprticle data rrays for use on the GPU
    gpu_pdata_arrays& d_pdata = m_pdata->acquireReadWriteGPU();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the second step of the integration on the GPU
    gpu_berendsen_step_two(d_pdata,
                           d_index_array.data,
                           group_size,
                           d_net_force.data,
                           m_block_size,
                           m_deltaT);

    // check if an error occurred
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // release the particle data
    m_pdata->release();

    if (m_prof)
        m_prof->pop();
    }

void export_BerendsenGPU()
    {
    class_<TwoStepBerendsenGPU, boost::shared_ptr<TwoStepBerendsenGPU>, bases<TwoStepBerendsen>, boost::noncopyable>
    ("TwoStepBerendsenGPU", init< boost::shared_ptr<SystemDefinition>,
                            boost::shared_ptr<ParticleGroup>,
                            boost::shared_ptr<ComputeThermo>,
                            Scalar,
                            boost::shared_ptr<Variant>
                            >())
    ;
    }

