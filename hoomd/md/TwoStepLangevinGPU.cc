/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

// Maintainer: joaander

#include "TwoStepLangevinGPU.h"
#include "TwoStepNVEGPU.cuh"
#include "TwoStepLangevinGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \file TwoStepLangevinGPU.h
    \brief Contains code for the TwoStepLangevinGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepLangevinGPU::TwoStepLangevinGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                       boost::shared_ptr<ParticleGroup> group,
                                       boost::shared_ptr<Variant> T,
                                       unsigned int seed,
                                       bool use_lambda,
                                       Scalar lambda,
                                       bool noiseless_t,
                                       bool noiseless_r,
                                       const std::string& suffix)
    : TwoStepLangevin(sysdef, group, T, seed, use_lambda, lambda, noiseless_t, noiseless_r, suffix)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepLangevinGPU while CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepLangevinGPU");
        }

    // allocate the sum arrays
    GPUArray<Scalar> sum(1, m_exec_conf);
    m_sum.swap(sum);

    // initialize the partial sum array
    m_block_size = 256;
    unsigned int group_size = m_group->getNumMembers();
    m_num_blocks = group_size / m_block_size + 1;
    GPUArray<Scalar> partial_sum1(m_num_blocks, m_exec_conf);
    m_partial_sum1.swap(partial_sum1);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.

    This method is copied directoy from TwoStepNVEGPU::integrateStepOne() and reimplemented here to avoid multiple.
*/
void TwoStepLangevinGPU::integrateStepOne(unsigned int timestep)
    {
    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "Langevin step 1");

    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getNumMembers();

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    // perform the update on the GPU
    gpu_nve_step_one(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_image.data,
                     d_index_array.data,
                     group_size,
                     box,
                     m_deltaT,
                     false,
                     0,
                     false,
                     256);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

        gpu_nve_angular_step_one(d_orientation.data,
                                 d_angmom.data,
                                 d_inertia.data,
                                 d_net_torque.data,
                                 d_index_array.data,
                                 group_size,
                                 m_deltaT,
                                 1.0);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepLangevinGPU::integrateStepTwo(unsigned int timestep)
    {
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "Langevin step 2");

    // get the dimensionality of the system
    const unsigned int D = m_sysdef->getNDimensions();

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma_r(m_gamma_r, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        {
        ArrayHandle<Scalar> d_partial_sumBD(m_partial_sum1, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_sumBD(m_sum, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

        unsigned int group_size = m_group->getNumMembers();
        m_num_blocks = group_size / m_block_size + 1;

        // perform the update on the GPU
        langevin_step_two_args args;
        args.d_gamma = d_gamma.data;
        args.n_types = m_gamma.getNumElements();
        args.use_lambda = m_use_lambda;
        args.lambda = m_lambda;
        args.T = m_T->getValue(timestep);
        args.timestep = timestep;
        args.seed = m_seed;
        args.d_sum_bdenergy = d_sumBD.data;
        args.d_partial_sum_bdenergy = d_partial_sumBD.data;
        args.block_size = m_block_size;
        args.num_blocks = m_num_blocks;
        args.noiseless_t = m_noiseless_t;
        args.noiseless_r = m_noiseless_r;
        args.tally = m_tally;

        gpu_langevin_step_two(d_pos.data,
                              d_vel.data,
                              d_accel.data,
                              d_diameter.data,
                              d_tag.data,
                              d_index_array.data,
                              group_size,
                              d_net_force.data,
                              args,
                              m_deltaT,
                              D);

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (m_aniso)
            {
            // second part of angular update
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

            unsigned int group_size = m_group->getNumMembers();
            gpu_langevin_angular_step_two(d_pos.data,
                                     d_orientation.data,
                                     d_angmom.data,
                                     d_inertia.data,
                                     d_net_torque.data,
                                     d_index_array.data,
                                     d_gamma_r.data,
                                     d_tag.data,
                                     group_size,
                                     args,
                                     m_deltaT,
                                     D,
                                     1.0
                                     );

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        }



    if (m_tally)
        {
        ArrayHandle<Scalar> h_sumBD(m_sum, access_location::host, access_mode::read);
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            MPI_Allreduce(MPI_IN_PLACE, &h_sumBD.data[0], 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            }
        #endif
        m_reservoir_energy -= h_sumBD.data[0]*m_deltaT;
        m_extra_energy_overdeltaT= 0.5*h_sumBD.data[0];
        }
    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_TwoStepLangevinGPU()
    {
    class_<TwoStepLangevinGPU, boost::shared_ptr<TwoStepLangevinGPU>, bases<TwoStepLangevin>, boost::noncopyable>
        ("TwoStepLangevinGPU", init< boost::shared_ptr<SystemDefinition>,
                               boost::shared_ptr<ParticleGroup>,
                               boost::shared_ptr<Variant>,
                               unsigned int,
                               bool,
                               Scalar,
                               bool,
                               bool,
                               const std::string&
                               >())
        ;
    }
