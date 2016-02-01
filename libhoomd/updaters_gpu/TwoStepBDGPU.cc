/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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

#include "TwoStepBDGPU.h"
#include "TwoStepBDGPU.cuh"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \file TwoStepBDGPU.h
    \brief Contains code for the TwoStepBDGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
*/
TwoStepBDGPU::TwoStepBDGPU(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<ParticleGroup> group,
                           boost::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda,
                           bool noiseless_t,
                           bool noiseless_r)
    : TwoStepBD(sysdef, group, T, seed, use_lambda, lambda, noiseless_t, noiseless_r)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepBDGPU while CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepBDGPU");
        }

    m_block_size = 256;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward a full time step and velocities are redrawn from the proper distrubtion.
*/
void TwoStepBDGPU::integrateStepOne(unsigned int timestep)
    {
    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "BD step 1");

    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getNumMembers();
    const unsigned int D = m_sysdef->getNDimensions();
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // for rotational noise
    ArrayHandle<Scalar> d_gamma_r(m_gamma_r, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::readwrite);
    
    unsigned int num_blocks = group_size / m_block_size + 1;

    langevin_step_two_args args;
    args.d_gamma = d_gamma.data;
    args.n_types = m_gamma.getNumElements();
    args.use_lambda = m_use_lambda;
    args.lambda = m_lambda;
    args.T = m_T->getValue(timestep);
    args.timestep = timestep;
    args.seed = m_seed;
    args.d_sum_bdenergy = NULL;
    args.d_partial_sum_bdenergy = NULL;
    args.block_size = m_block_size;
    args.num_blocks = num_blocks;
    args.tally = false;
    
    bool aniso = m_aniso;

    // perform the update on the GPU
    gpu_brownian_step_one(d_pos.data,
                          d_vel.data,
                          d_image.data,
                          box,
                          d_diameter.data,
                          d_tag.data,
                          d_index_array.data,
                          group_size,
                          d_net_force.data,
                          d_gamma_r.data,
                          d_orientation.data,
                          d_torque.data,
                          args,
                          aniso,
                          m_deltaT,
                          D, 
                          m_noiseless_t, 
                          m_noiseless_r);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepBDGPU::integrateStepTwo(unsigned int timestep)
    {
    // there is no step 2
    }

void export_TwoStepBDGPU()
    {
    class_<TwoStepBDGPU, boost::shared_ptr<TwoStepBDGPU>, bases<TwoStepBD>, boost::noncopyable>
        ("TwoStepBDGPU", init< boost::shared_ptr<SystemDefinition>,
                               boost::shared_ptr<ParticleGroup>,
                               boost::shared_ptr<Variant>,
                               unsigned int,
                               bool,
                               Scalar,
                               bool,
                               bool>())
        ;
    }
