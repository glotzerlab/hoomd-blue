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

#include "TwoStepNPTRigidGPU.h"
#include "TwoStepNPTRigidGPU.cuh"
#include <boost/python.hpp>
#include <boost/bind.hpp>

using namespace std;
using namespace boost::python;
using namespace boost;

/*! \file TwoStepNPTRigidGPU.cc
    \brief Contains code for the TwoStepNPTRigidGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param thermo_all ComputeThermo to compute the pressure of the entire system
    \param suffix Suffix to attach to the end of log quantity names
    \param tau Time constant for thermostat
    \param tauP Time constant for barostat
    \param T Controlled temperature
    \param P Controlled pressure
    \param couple Coupling mode
    \param flags Barostatted simulation box degrees of freedom
    \param tchain Number of thermostats in the thermostat chain
    \param pchain Number of thermostats coupled with the barostat
    \param iter Number of inner iterations to update the thermostats
*/
TwoStepNPTRigidGPU::TwoStepNPTRigidGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                   boost::shared_ptr<ParticleGroup> group,
                                   boost::shared_ptr<ComputeThermo> thermo_group,
                                   boost::shared_ptr<ComputeThermo> thermo_all,
                                   const std::string& suffix,
                                   Scalar tau,
                                   Scalar tauP,
                                   boost::shared_ptr<Variant> T,
                                   boost::shared_ptr<Variant> P,
                                   couplingMode couple,
                                   unsigned int flags,
                                   unsigned int tchain,
                                   unsigned int pchain,
                                   unsigned int iter)
    : TwoStepNPTRigid(sysdef, group, thermo_group, thermo_all, suffix, tau, tauP, T, P, couple, flags, tchain, pchain, iter)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepNPTRigidGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing TwoStepNPTRigidGPU");
        }

    // allocate the total sum variables
    GPUArray<Scalar> sum2K(1, m_pdata->getExecConf());
    m_sum2K.swap(sum2K);
    GPUArray<Scalar> sumW(1, m_pdata->getExecConf());
    m_sumW.swap(sumW);
    GPUArray<Scalar> sum_virial_rigid(1, m_pdata->getExecConf());
    m_sum_virial_rigid.swap(sum_virial_rigid);

    // initialize the partial sum2K array
    m_block_size = 128;
    m_group_num_blocks = m_group->getNumMembers() / m_block_size + 1;
    m_full_num_blocks = m_pdata->getN() / m_block_size + 1;
    GPUArray<Scalar> partial_sum2K(m_full_num_blocks, m_pdata->getExecConf());
    m_partial_sum2K.swap(partial_sum2K);
    GPUArray<Scalar> partial_sumW(m_full_num_blocks, m_pdata->getExecConf());
    m_partial_sumW.swap(partial_sumW);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepNPTRigidGPU::integrateStepOne(unsigned int timestep)
    {
    if (m_first_step)
        {
        setup();

        GPUArray<Scalar> partial_Ksum_t(m_n_bodies, m_pdata->getExecConf());
        m_partial_Ksum_t.swap(partial_Ksum_t);
        GPUArray<Scalar> partial_Ksum_r(m_n_bodies, m_pdata->getExecConf());
        m_partial_Ksum_r.swap(partial_Ksum_r);
        GPUArray<Scalar> Ksum_t(1, m_pdata->getExecConf());
        m_Ksum_t.swap(Ksum_t);
        GPUArray<Scalar> Ksum_r(1, m_pdata->getExecConf());
        m_Ksum_r.swap(Ksum_r);
        GPUArray<Scalar4> new_box(1, m_pdata->getExecConf());
        m_new_box.swap(new_box);

        m_first_step = false;
        }

    // sanity check
    if (m_n_bodies <= 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push("NPT rigid step 1");

    { // request handles to GPU arrays

    Scalar tmp, scale_r;
    Scalar3 scale_t, scale_v;

    // velocity scaling factors from the thermostat chain
    scale_t.x = scale_t.y = scale_t.z = exp(-m_dt_half * m_eta_dot_t[0]);
    scale_r = exp(-m_dt_half * m_eta_dot_r[0]);

    // and from the thermostat chain coupled with barostat
    scale_t.x *= exp(-m_dt_half * (m_epsilon_dot[0] + m_mtk_term2));
    scale_t.y *= exp(-m_dt_half * (m_epsilon_dot[3] + m_mtk_term2));
    scale_t.z *= exp(-m_dt_half * (m_epsilon_dot[5] + m_mtk_term2));
    scale_r *= exp(-m_dt_half * m_pdim * m_mtk_term2);

    // velocity scaling factors from barostat
    tmp = m_dt_half * m_epsilon_dot[0];
    scale_v.x = m_deltaT * exp(tmp) * maclaurin_series(tmp);
    tmp = m_dt_half * m_epsilon_dot[3];
    scale_v.y = m_deltaT * exp(tmp) * maclaurin_series(tmp);
    tmp = m_dt_half * m_epsilon_dot[5];
    scale_v.z = m_deltaT * exp(tmp) * maclaurin_series(tmp);

    // box scaling factors
    Scalar4 dilation;
    dilation.x = exp(m_deltaT * m_epsilon_dot[0]);
    dilation.y = exp(m_deltaT * m_epsilon_dot[3]);
    dilation.z = exp(m_deltaT * m_epsilon_dot[5]);

    m_akin_t = m_akin_r = Scalar(0.0);

    // access all the needed data
    BoxDim box = m_pdata->getBox();
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
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
    d_rdata.particle_orientation = d_particle_orientation.data;

    ArrayHandle<Scalar> partial_Ksum_t_handle(m_partial_Ksum_t, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> partial_Ksum_r_handle(m_partial_Ksum_r, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> Ksum_t_handle(m_Ksum_t, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> Ksum_r_handle(m_Ksum_r, access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> new_box_handle(m_new_box, access_location::device, access_mode::readwrite);

    gpu_npt_rigid_data d_npt_rdata;
    d_npt_rdata.n_bodies = m_n_bodies;
    d_npt_rdata.nf_t = m_nf_t;
    d_npt_rdata.nf_r = m_nf_r;
    d_npt_rdata.dimension = m_dimension;
    d_npt_rdata.scale_t = make_scalar4(scale_t.x,scale_t.y,scale_t.z,Scalar(0.0));
    d_npt_rdata.scale_r = scale_r;
    d_npt_rdata.scale_v = make_scalar4(scale_v.x,scale_v.y,scale_v.z,Scalar(0.0));
    d_npt_rdata.partial_Ksum_t = partial_Ksum_t_handle.data;
    d_npt_rdata.partial_Ksum_r = partial_Ksum_r_handle.data;
    d_npt_rdata.Ksum_t = Ksum_t_handle.data;
    d_npt_rdata.Ksum_r = Ksum_r_handle.data;
    d_npt_rdata.new_box = new_box_handle.data;
    d_npt_rdata.dilation = dilation;

    // perform the first-half integration on the GPU
    // determine the new box sizes
    // remap the bodies to the new box
    gpu_npt_rigid_step_one(d_rdata,
                           d_index_array.data,
                           group_size,
                           d_net_force.data,
                           box,
                           d_npt_rdata,
                           m_deltaT);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // tally the body kinetic energies
    gpu_npt_rigid_reduce_ksum(d_npt_rdata);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    } // release handles to body arrays for next step

    // set new box
    {
    ArrayHandle<Scalar4> new_box_handle(m_new_box, access_location::host, access_mode::read);
    Scalar Lx = new_box_handle.data[0].x;
    Scalar Ly = new_box_handle.data[0].y;
    Scalar Lz = new_box_handle.data[0].z;
    m_pdata->setGlobalBoxL(make_scalar3(Lx, Ly, Lz));
    }

    // update barostat
    {
    ArrayHandle<Scalar> Ksum_t_handle(m_Ksum_t, access_location::host, access_mode::read);
    ArrayHandle<Scalar> Ksum_r_handle(m_Ksum_r, access_location::host, access_mode::read);

    m_akin_t = Ksum_t_handle.data[0];
    m_akin_r = Ksum_r_handle.data[0];

    // compute thermostat chain coupled with thermostat
    update_nh_tchain(m_akin_t, m_akin_r, timestep);

    // compute target pressure
    compute_target_pressure(timestep);

    // compute thermostat chain coupled with barotat
    update_nh_pchain(timestep);
    }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNPTRigidGPU::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push("NPT rigid step 2");

    { // request handles to GPU arrays

    Scalar scale_r;
    Scalar3 scale_t;

    // velocity scaling factors from the thermostat chain
    scale_t.x = scale_t.y = scale_t.z = exp(-m_dt_half * m_eta_dot_t[0]);
    scale_r = exp(-m_dt_half * m_eta_dot_r[0]);

    // and from the thermostat chain coupled with barostat
    scale_t.x *= exp(-m_dt_half * (m_epsilon_dot[0] + m_mtk_term2));
    scale_t.y *= exp(-m_dt_half * (m_epsilon_dot[3] + m_mtk_term2));
    scale_t.z *= exp(-m_dt_half * (m_epsilon_dot[5] + m_mtk_term2));
    scale_r *= exp(-m_dt_half * m_pdim * m_mtk_term2);

    m_akin_t = m_akin_r = Scalar(0.0);

    BoxDim box = m_pdata->getBox();
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    const GPUArray< Scalar4 >& net_torque = m_pdata->getNetTorqueArray();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_torque(net_torque, access_location::device, access_mode::read);
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
    d_rdata.particle_orientation = d_particle_orientation.data;

    ArrayHandle<Scalar> partial_Ksum_t_handle(m_partial_Ksum_t, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> partial_Ksum_r_handle(m_partial_Ksum_r, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> Ksum_t_handle(m_Ksum_t, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> Ksum_r_handle(m_Ksum_r, access_location::device, access_mode::readwrite);

    gpu_npt_rigid_data d_npt_rdata;
    d_npt_rdata.n_bodies = m_n_bodies;
    d_npt_rdata.nf_t = m_nf_t;
    d_npt_rdata.nf_r = m_nf_r;
    d_npt_rdata.dimension = m_dimension;
    d_npt_rdata.scale_t = make_scalar4(scale_t.x,scale_t.y,scale_t.z,Scalar(0.0));
    d_npt_rdata.scale_r = scale_r;
    d_npt_rdata.partial_Ksum_t = partial_Ksum_t_handle.data;
    d_npt_rdata.partial_Ksum_r = partial_Ksum_r_handle.data;
    d_npt_rdata.Ksum_t = Ksum_t_handle.data;
    d_npt_rdata.Ksum_r = Ksum_r_handle.data;

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
    gpu_npt_rigid_step_two(d_rdata,
                           d_index_array.data,
                           group_size,
                           d_net_force.data,
                           d_net_virial.data,
                           box,
                           d_npt_rdata,
                           m_deltaT);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    gpu_npt_rigid_reduce_ksum(d_npt_rdata);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    } // release handles to GPU arrays for access in the next step

    // update barostat
    {
    ArrayHandle<Scalar> Ksum_t_handle(m_Ksum_t, access_location::host, access_mode::read);
    ArrayHandle<Scalar> Ksum_r_handle(m_Ksum_r, access_location::host, access_mode::read);

    m_akin_t = Ksum_t_handle.data[0];
    m_akin_r = Ksum_r_handle.data[0];

    // compute current pressure
    compute_current_pressure(timestep);

    // compute target pressure
    compute_target_pressure(timestep);

    // compute barostat
    update_nh_barostat(m_akin_t, m_akin_r);
    }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepNPTRigidGPU()
    {
    class_<TwoStepNPTRigidGPU, boost::shared_ptr<TwoStepNPTRigidGPU>, bases<TwoStepNPTRigid>, boost::noncopyable>
        ("TwoStepNPTRigidGPU", init< boost::shared_ptr<SystemDefinition>,
        boost::shared_ptr<ParticleGroup>,
        boost::shared_ptr<ComputeThermo>,
        boost::shared_ptr<ComputeThermo>,
        const std::string&,
        Scalar,
        Scalar,
        boost::shared_ptr<Variant>,
        boost::shared_ptr<Variant>,
        TwoStepNHRigid::couplingMode,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int >())
        ;
    }

