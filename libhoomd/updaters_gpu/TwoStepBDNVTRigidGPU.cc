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

// Maintainer: ndtrung

#include "TwoStepBDNVTRigidGPU.h"
#include "TwoStepNVERigidGPU.cuh"
#include "TwoStepLangevinGPU.cuh"
#include "TwoStepBDNVTRigidGPU.cuh"
#include <boost/python.hpp>
#include <boost/bind.hpp>

using namespace std;
using namespace boost::python;
using namespace boost;

/*! \file TwoStepBDNVTRigidGPU.h
    \brief Contains code for the TwoStepBDNVTRigidPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param suffix Suffix to attach to the end of log quantity names
    \param T Controlled temperature
    \param seed An interger seed for random number generator
    \param gamma_diam Flag for using particle diameter-dependent gamma
*/
TwoStepBDNVTRigidGPU::TwoStepBDNVTRigidGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                        boost::shared_ptr<ParticleGroup> group,
                                        boost::shared_ptr<Variant> T,
                                        unsigned int seed,
                                        bool gamma_diam,
                                        bool noiseless_t,
                                        bool noiseless_r)
    : TwoStepBDNVTRigid(sysdef, group, T, seed, gamma_diam, noiseless_t, noiseless_r)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepBDNVTRigidGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing TwoStepBDNVTRigidGPU");
        }
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepBDNVTRigidGPU::integrateStepOne(unsigned int timestep)
    {
    if (m_first_step)
        {
        setup();
        m_first_step = false;
        }

    // sanity check
    if (m_n_bodies <= 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push("BD NVT rigid step 1");

    // access to the force and virial
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body_index_array(m_body_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();

    // get the rigid data from SystemDefinition
    boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();

    ArrayHandle<Scalar> body_mass_handle(rigid_data->getBodyMass(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> moment_inertia_handle(rigid_data->getMomentInertia(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(rigid_data->getAngVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> orientation_handle(rigid_data->getOrientation(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> body_image_handle(rigid_data->getBodyImage(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> particle_pos_handle(rigid_data->getParticlePos(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> particle_indices_handle(rigid_data->getParticleIndices(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> force_handle(rigid_data->getForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(rigid_data->getTorque(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> conjqm_handle(rigid_data->getConjqm(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> particle_oldpos_handle(rigid_data->getParticleOldPos(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> particle_oldvel_handle(rigid_data->getParticleOldVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_particle_offset(rigid_data->getParticleOffset(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_particle_orientation(rigid_data->getParticleOrientation(), access_location::device, access_mode::read);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = rigid_data->getNumBodies();
    d_rdata.n_group_bodies = m_n_bodies;
    d_rdata.nmax = rigid_data->getNmax();
    d_rdata.local_beg = 0;
    d_rdata.local_num = m_n_bodies;

    d_rdata.body_indices = d_body_index_array.data;
    d_rdata.body_mass = body_mass_handle.data;
    d_rdata.moment_inertia = moment_inertia_handle.data;
    d_rdata.com = com_handle.data;
    d_rdata.vel = vel_handle.data;
    d_rdata.angvel = angvel_handle.data;
    d_rdata.angmom = angmom_handle.data;
    d_rdata.orientation = orientation_handle.data;
    d_rdata.body_image = body_image_handle.data;
    d_rdata.particle_pos = particle_pos_handle.data;
    d_rdata.particle_indices = particle_indices_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;
    d_rdata.conjqm = conjqm_handle.data;
    d_rdata.particle_oldpos = particle_oldpos_handle.data;
    d_rdata.particle_oldvel = particle_oldvel_handle.data;
    d_rdata.particle_offset = d_particle_offset.data;
    d_rdata.particle_orientation = d_particle_orientation.data;

    Scalar D = Scalar(m_sysdef->getNDimensions());

    // perform the update on the GPU
    gpu_bdnvt_rigid_step_one(d_rdata,
                           d_index_array.data,
                           group_size,
                           d_net_force.data,
                           box,
                           m_deltaT, 
                           m_gamma_r,
                           D);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepBDNVTRigidGPU::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push("BD NVT rigid step 2");

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    const GPUArray< Scalar4 >& net_torque = m_pdata->getNetTorqueArray();

    // get the dimensionality of the system
    const Scalar D = Scalar(m_sysdef->getNDimensions());

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_torque(net_torque, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body_index_array(m_body_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();

    // get the rigid data from SystemDefinition
    boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();

    ArrayHandle<Scalar> body_mass_handle(rigid_data->getBodyMass(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> moment_inertia_handle(rigid_data->getMomentInertia(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(rigid_data->getAngVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> orientation_handle(rigid_data->getOrientation(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> particle_pos_handle(rigid_data->getParticlePos(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> particle_indices_handle(rigid_data->getParticleIndices(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> force_handle(rigid_data->getForce(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> torque_handle(rigid_data->getTorque(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> conjqm_handle(rigid_data->getConjqm(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> particle_oldpos_handle(rigid_data->getParticleOldPos(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> particle_oldvel_handle(rigid_data->getParticleOldVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_particle_offset(rigid_data->getParticleOffset(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_particle_orientation(rigid_data->getParticleOrientation(), access_location::device, access_mode::read);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = rigid_data->getNumBodies();
    d_rdata.n_group_bodies = m_n_bodies;
    d_rdata.nmax = rigid_data->getNmax();
    d_rdata.local_beg = 0;
    d_rdata.local_num = m_n_bodies;

    d_rdata.body_indices = d_body_index_array.data;
    d_rdata.body_mass = body_mass_handle.data;
    d_rdata.moment_inertia = moment_inertia_handle.data;
    d_rdata.com = com_handle.data;
    d_rdata.vel = vel_handle.data;
    d_rdata.angvel = angvel_handle.data;
    d_rdata.angmom = angmom_handle.data;
    d_rdata.orientation = orientation_handle.data;
    d_rdata.particle_pos = particle_pos_handle.data;
    d_rdata.particle_indices = particle_indices_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;
    d_rdata.conjqm = conjqm_handle.data;
    d_rdata.particle_oldpos = particle_oldpos_handle.data;
    d_rdata.particle_oldvel = particle_oldvel_handle.data;
    d_rdata.particle_offset = d_particle_offset.data;
    d_rdata.particle_orientation = d_particle_orientation.data;

    // compute the Langevin forces on the GPU
    langevin_step_two_args args;
    args.d_gamma = d_gamma.data;
    args.n_types = m_gamma.getNumElements();
    args.use_lambda = m_gamma_diam;
    args.lambda = Scalar(1.0);
    args.T = m_temperature->getValue(timestep);
    args.timestep = timestep;
    args.seed = m_seed;

    gpu_bdnvt_force(d_pos.data,
                    d_vel.data,
                    d_diameter.data,
                    d_tag.data,
                    d_index_array.data,
                    group_size,
                    d_net_force.data,
                    args,
                    m_deltaT,
                    D);


    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    gpu_rigid_force(d_rdata,
                    d_index_array.data,
                    group_size,
                    d_net_force.data,
                    d_net_torque.data,
                    box,
                    m_deltaT);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // perform the update on the GPU
    gpu_nve_rigid_step_two(d_rdata,
                           d_index_array.data,
                           group_size,
                           d_net_force.data,
                           d_net_virial.data,
                           box,
                           m_deltaT);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepBDNVTRigidGPU()
    {
    class_<TwoStepBDNVTRigidGPU, boost::shared_ptr<TwoStepBDNVTRigidGPU>, bases<TwoStepBDNVTRigid>, boost::noncopyable>
        ("TwoStepBDNVTRigidGPU", init< boost::shared_ptr<SystemDefinition>,
            boost::shared_ptr<ParticleGroup>,
            boost::shared_ptr<Variant>,
            unsigned int,
            bool,
            bool,
            bool >())
        ;
    }
