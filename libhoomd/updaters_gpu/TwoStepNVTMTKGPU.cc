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

// Maintainer: jglaser

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include "TwoStepNVTMTKGPU.h"
#include "TwoStepNVTMTKGPU.cuh"
#include "TwoStepNPTMTKGPU.cuh"

#ifdef ENABLE_MPI
#include "Communicator.h"
#include "HOOMDMPI.h"
#endif

/*! \file TwoStepNVTMTKGPU.h
    \brief Contains code for the TwoStepNVTMTKGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepNVTMTKGPU::TwoStepNVTMTKGPU(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<ParticleGroup> group,
                             boost::shared_ptr<ComputeThermo> thermo,
                             Scalar tau,
                             boost::shared_ptr<Variant> T,
                             const std::string& suffix)
    : TwoStepNVTMTK(sysdef, group, thermo, tau, T, suffix)
    {
    // only one GPU is supported
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepNVTMTKPU when CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepNVTMTKGPU");
        }

    // initialize autotuner
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        valid_params.push_back(block_size);

    m_tuner_one.reset(new Autotuner(valid_params, 5, 100000, "nvt_mtk_step_one", this->m_exec_conf));
    m_tuner_two.reset(new Autotuner(valid_params, 5, 100000, "nvt_mtk_step_two", this->m_exec_conf));

    m_reduction_block_size = 512;

    // this breaks memory scaling (calculate memory requirements from global group size)
    // unless we reallocate memory with every change of the maximum particle number
    m_num_blocks = m_group->getNumMembersGlobal() / m_reduction_block_size + 1;
    GPUArray< Scalar > scratch(m_num_blocks, exec_conf);
    m_scratch.swap(scratch);

    GPUArray< Scalar> temperature(1, exec_conf, true);
    m_temperature.swap(temperature);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Nose-Hoover method
*/
void TwoStepNVTMTKGPU::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NVT MTK step 1");

    // compute the current thermodynamic properties
    m_thermo->compute(timestep);

    // compute temperature for the next half time step
    m_curr_T = m_thermo->getTemperature();

    // advance thermostat
    advanceThermostat(timestep, false);

    // access all the needed data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    BoxDim box = m_pdata->getBox();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the update on the GPU
    m_tuner_one->begin();
    gpu_nvt_mtk_step_one(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_image.data,
                     d_index_array.data,
                     group_size,
                     box,
                     m_tuner_one->getParam(),
                     m_exp_thermo_fac,
                     m_deltaT);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_one->end();

    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNVTMTKGPU::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NVT MTK step 2");

    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the update on the GPU
    m_tuner_two->begin();
    gpu_nvt_mtk_step_two(d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     group_size,
                     d_net_force.data,
                     m_tuner_two->getParam(),
                     m_deltaT);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_two->end();

        {
        // recalulate temperature
        ArrayHandle<Scalar> d_temperature(m_temperature, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch(m_scratch, access_location::device, access_mode::overwrite);

        // update number of blocks to current group size
        m_num_blocks = m_group->getNumMembers() / m_reduction_block_size + 1;

        gpu_npt_mtk_temperature(d_temperature.data,
                                d_vel.data,
                                d_scratch.data,
                                m_num_blocks,
                                m_reduction_block_size,
                                d_index_array.data,
                                group_size,
                                m_thermo->getNDOF());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // read back intermediate temperature from GPU
        {
        ArrayHandle<Scalar> h_temperature(m_temperature, access_location::host, access_mode::read);
        m_curr_T= *h_temperature.data;
        }

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &m_curr_T, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator() );
        }
    #endif


    // get temperature and advance thermostat
    advanceThermostat(timestep+1,true);

    // rescale velocities
    gpu_npt_mtk_thermostat(d_vel.data,
                           d_index_array.data,
                           group_size,
                           m_exp_thermo_fac);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }

void export_TwoStepNVTMTKGPU()
    {
    class_<TwoStepNVTMTKGPU, boost::shared_ptr<TwoStepNVTMTKGPU>, bases<TwoStepNVTMTK>, boost::noncopyable>
        ("TwoStepNVTMTKGPU", init< boost::shared_ptr<SystemDefinition>,
                          boost::shared_ptr<ParticleGroup>,
                          boost::shared_ptr<ComputeThermo>,
                          Scalar,
                          boost::shared_ptr<Variant>,
                          const std::string&
                          >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
