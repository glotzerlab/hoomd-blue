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
// Maintainer: ndtrung

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include "TwoStepNVTRigidGPU.h"
#include "TwoStepNVTRigidGPU.cuh"

/*! \file TwoStepNVTRigidGPU.cc
    \brief Contains code for the TwoStepNVTRigidGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param T Controlled temperature
    \param tau Time constant
*/
TwoStepNVTRigidGPU::TwoStepNVTRigidGPU(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<ParticleGroup> group,
                             boost::shared_ptr<ComputeThermo> thermo,  
                             boost::shared_ptr<Variant> T,
                             Scalar tau)
    : TwoStepNVTRigid(sysdef, group, thermo, T, tau)
    {
    // only one GPU is supported
    if (exec_conf.gpu.size() != 1)
        {
        cerr << endl << "***Error! Creating a TwoStepNVTRigidGPU with 0 or more than one GPUs" << endl << endl;
        throw std::runtime_error("Error initializing TwoStepNVTRigidGPU");
        }
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepNVTRigidGPU::integrateStepOne(unsigned int timestep)
    {
    if (m_first_step)
        {
        setup();
        
        // sanity check
        if (m_n_bodies <= 0)
            return;
        
        GPUArray<float> partial_Ksum_t(m_n_bodies, m_pdata->getExecConf());
        m_partial_Ksum_t.swap(partial_Ksum_t);
        
        GPUArray<float> partial_Ksum_r(m_n_bodies, m_pdata->getExecConf());
        m_partial_Ksum_r.swap(partial_Ksum_r);
        
        GPUArray<float> Ksum_t(1, m_pdata->getExecConf());
        m_Ksum_t.swap(Ksum_t);
        
        GPUArray<float> Ksum_r(1, m_pdata->getExecConf());
        m_Ksum_r.swap(Ksum_r);
        
        m_first_step = false;
        }
        
    // sanity check
    if (m_n_bodies <= 0)
        return;
        
    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NVT rigid step 1");
    
    // access all the needed data
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
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
    ArrayHandle<Scalar4> ex_space_handle(rigid_data->getExSpace(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> ey_space_handle(rigid_data->getEySpace(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> ez_space_handle(rigid_data->getEzSpace(), access_location::device, access_mode::readwrite);
    ArrayHandle<int> body_imagex_handle(rigid_data->getBodyImagex(), access_location::device, access_mode::readwrite);
    ArrayHandle<int> body_imagey_handle(rigid_data->getBodyImagey(), access_location::device, access_mode::readwrite);
    ArrayHandle<int> body_imagez_handle(rigid_data->getBodyImagez(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> particle_pos_handle(rigid_data->getParticlePos(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> particle_indices_handle(rigid_data->getParticleIndices(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> force_handle(rigid_data->getForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(rigid_data->getTorque(), access_location::device, access_mode::read);
    
    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = rigid_data->getNumBodies();
    d_rdata.n_group_bodies = m_n_bodies;
    d_rdata.nmax = rigid_data->getNmax();
    d_rdata.local_beg = 0;
    d_rdata.local_num = m_n_bodies;
    
    d_rdata.body_mass = body_mass_handle.data;
    d_rdata.moment_inertia = moment_inertia_handle.data;
    d_rdata.com = com_handle.data;
    d_rdata.vel = vel_handle.data;
    d_rdata.angvel = angvel_handle.data;
    d_rdata.angmom = angmom_handle.data;
    d_rdata.orientation = orientation_handle.data;
    d_rdata.ex_space = ex_space_handle.data;
    d_rdata.ey_space = ey_space_handle.data;
    d_rdata.ez_space = ez_space_handle.data;
    d_rdata.body_imagex = body_imagex_handle.data;
    d_rdata.body_imagey = body_imagey_handle.data;
    d_rdata.body_imagez = body_imagez_handle.data;
    d_rdata.particle_pos = particle_pos_handle.data;
    d_rdata.particle_indices = particle_indices_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;
    
    ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::read);
    ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> conjqm_handle(conjqm, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> partial_Ksum_t_handle(m_partial_Ksum_t, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> partial_Ksum_r_handle(m_partial_Ksum_r, access_location::device, access_mode::readwrite);
    
    gpu_nvt_rigid_data d_nvt_rdata;
    d_nvt_rdata.n_bodies = d_rdata.n_bodies;
    d_nvt_rdata.eta_dot_t0 = eta_dot_t_handle.data[0];
    d_nvt_rdata.eta_dot_r0 = eta_dot_r_handle.data[0];
    d_nvt_rdata.conjqm = conjqm_handle.data;
    d_nvt_rdata.partial_Ksum_t = partial_Ksum_t_handle.data;
    d_nvt_rdata.partial_Ksum_r = partial_Ksum_r_handle.data;
    
    // perform the update on the GPU
    exec_conf.tagAll(__FILE__, __LINE__);    
    exec_conf.gpu[0]->call(bind(gpu_nvt_rigid_step_one, 
                                d_pdata[0],
                                d_rdata, 
                                d_index_array.data,
                                group_size,
                                box,
                                d_nvt_rdata,
                                m_deltaT));

    
    m_pdata->release();
    
    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }
        
/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNVTRigidGPU::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;
        
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    
    // phase 1, reduce to find the final Ksum_t and Ksum_r
        {
        if (m_prof)
            m_prof->push(exec_conf, "NVT reducing");
        
        ArrayHandle<Scalar> partial_Ksum_t_handle(m_partial_Ksum_t, access_location::device, access_mode::read);
        ArrayHandle<Scalar> partial_Ksum_r_handle(m_partial_Ksum_r, access_location::device, access_mode::read);
        ArrayHandle<Scalar> Ksum_t_handle(m_Ksum_t, access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> Ksum_r_handle(m_Ksum_r, access_location::device, access_mode::readwrite);
    
        gpu_nvt_rigid_data d_nvt_rdata;
        d_nvt_rdata.n_bodies = m_sysdef->getRigidData()->getNumBodies();
        d_nvt_rdata.partial_Ksum_t = partial_Ksum_t_handle.data;
        d_nvt_rdata.partial_Ksum_r = partial_Ksum_r_handle.data;
        d_nvt_rdata.Ksum_t = Ksum_t_handle.data;
        d_nvt_rdata.Ksum_r = Ksum_r_handle.data;

        exec_conf.gpu[0]->call(bind(gpu_nvt_rigid_reduce_ksum, d_nvt_rdata));
        
        if (m_prof)
            m_prof->pop(exec_conf);
        }
    
    // phase 1.5, move the thermostat variables forward
        {
        ArrayHandle<Scalar> Ksum_t_handle(m_Ksum_t, access_location::host, access_mode::read);
        ArrayHandle<Scalar> Ksum_r_handle(m_Ksum_r, access_location::host, access_mode::read);
            
        Scalar Ksum_t = Ksum_t_handle.data[0];
        Scalar Ksum_r = Ksum_r_handle.data[0];
        update_nhcp(Ksum_t, Ksum_r, m_deltaT);
        }

    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NVT rigid step 2");
    
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
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
    ArrayHandle<Scalar4> ex_space_handle(rigid_data->getExSpace(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> ey_space_handle(rigid_data->getEySpace(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> ez_space_handle(rigid_data->getEzSpace(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> particle_pos_handle(rigid_data->getParticlePos(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> particle_indices_handle(rigid_data->getParticleIndices(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> force_handle(rigid_data->getForce(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> torque_handle(rigid_data->getTorque(), access_location::device, access_mode::readwrite);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = rigid_data->getNumBodies();
    d_rdata.n_group_bodies = m_n_bodies;
    d_rdata.nmax = rigid_data->getNmax();
    d_rdata.local_beg = 0;
    d_rdata.local_num = m_n_bodies;
    
    d_rdata.body_mass = body_mass_handle.data;
    d_rdata.moment_inertia = moment_inertia_handle.data;
    d_rdata.com = com_handle.data;
    d_rdata.vel = vel_handle.data;
    d_rdata.angvel = angvel_handle.data;
    d_rdata.angmom = angmom_handle.data;
    d_rdata.ex_space = ex_space_handle.data;
    d_rdata.ey_space = ey_space_handle.data;
    d_rdata.ez_space = ez_space_handle.data;
    d_rdata.orientation = orientation_handle.data;
    d_rdata.particle_pos = particle_pos_handle.data;
    d_rdata.particle_indices = particle_indices_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;

    ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::read);
    ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> conjqm_handle(conjqm, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> partial_Ksum_t_handle(m_partial_Ksum_t, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> partial_Ksum_r_handle(m_partial_Ksum_r, access_location::device, access_mode::readwrite);
    
    gpu_nvt_rigid_data d_nvt_rdata;
    d_nvt_rdata.n_bodies = d_rdata.n_bodies;
    d_nvt_rdata.eta_dot_t0 = eta_dot_t_handle.data[0];
    d_nvt_rdata.eta_dot_r0 = eta_dot_r_handle.data[0];
    d_nvt_rdata.conjqm = conjqm_handle.data;
    d_nvt_rdata.partial_Ksum_t = partial_Ksum_t_handle.data;
    d_nvt_rdata.partial_Ksum_r = partial_Ksum_r_handle.data;
    
    exec_conf.tagAll(__FILE__, __LINE__);
    exec_conf.gpu[0]->call(bind(gpu_rigid_force, 
                                d_pdata[0], 
                                d_rdata, 
                                d_index_array.data,
                                group_size,
                                d_net_force.data,
                                box, 
                                m_deltaT)); 
                                
    // perform the update on the GPU
    exec_conf.tagAll(__FILE__, __LINE__);
    exec_conf.gpu[0]->call(bind(gpu_nvt_rigid_step_two, 
                                d_pdata[0], 
                                d_rdata, 
                                d_index_array.data,
                                group_size,
                                d_net_force.data,
                                box,
                                d_nvt_rdata, 
                                m_deltaT)); 
                                
   
    m_pdata->release();
    
    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }

void export_TwoStepNVTRigidGPU()
    {
    class_<TwoStepNVTRigidGPU, boost::shared_ptr<TwoStepNVTRigidGPU>, bases<TwoStepNVTRigid>, boost::noncopyable>
        ("TwoStepNVTRigidGPU", init< boost::shared_ptr<SystemDefinition>, 
        boost::shared_ptr<ParticleGroup>, 
        boost::shared_ptr<ComputeThermo>, 
        boost::shared_ptr<Variant> >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

