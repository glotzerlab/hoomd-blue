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

#define EPSILON 1e-6

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include "FIREEnergyMinimizerRigidGPU.h"
#include "FIREEnergyMinimizerRigidGPU.cuh"
#include "FIREEnergyMinimizerGPU.cuh"
#include "TwoStepNVERigidGPU.h"

/*! \file FIREEnergyMinimizerRigidGPU.h
    \brief Contains code for the FIREEnergyMinimizerRigidGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on (group is a placeholder for now)
    \param dt Time step for MD integrator
    \param reset_and_create_integrator Flag to indicate if resetting and creating integrator are needed 
    \post The method is constructed with the given particle data and a NULL profiler.
*/
FIREEnergyMinimizerRigidGPU::FIREEnergyMinimizerRigidGPU(boost::shared_ptr<SystemDefinition> sysdef, 
                                                        boost::shared_ptr<ParticleGroup> group, 
                                                        Scalar dt, 
                                                        bool reset_and_create_integrator)
    :   FIREEnergyMinimizerRigid(sysdef, group, dt, false) 
    {    
    // only one GPU is supported
    if (exec_conf.gpu.size() != 1)
        {
        cerr << endl << "***Error! Creating a FIREEnergyMinimizerRigidGPU with 0 or more than one GPUs" << endl << endl;
        throw std::runtime_error("Error initializing FIREEnergyMinimizerRigidGPU");
        }
    
    // allocate the sum arrays
    GPUArray<float> sum_pe(1, m_pdata->getExecConf());
    m_sum_pe.swap(sum_pe);
    GPUArray<float> sum_Pt(3, m_pdata->getExecConf());
    m_sum_Pt.swap(sum_Pt);
    GPUArray<float> sum_Pr(3, m_pdata->getExecConf());
    m_sum_Pr.swap(sum_Pr);
    
    m_block_size = 256; 
    m_num_blocks = m_nparticles / m_block_size + 1;
    GPUArray<float> partial_sum_pe(m_num_blocks, m_pdata->getExecConf());
    m_partial_sum_pe.swap(partial_sum_pe);
    
    if (reset_and_create_integrator)
        {
        reset();
    //    createIntegrator();
        boost::shared_ptr<TwoStepNVERigidGPU> integrator(new TwoStepNVERigidGPU(sysdef, group));
        addIntegrationMethod(integrator);
        setDeltaT(m_deltaT);
        }
    }

void FIREEnergyMinimizerRigidGPU::createIntegrator()
    {
    boost::shared_ptr<ParticleSelector> selector_rigid(new ParticleSelectorRigid(m_sysdef, true));
    boost::shared_ptr<ParticleGroup> group_rigid(new ParticleGroup(m_sysdef, selector_rigid));
    boost::shared_ptr<TwoStepNVERigidGPU> integrator(new TwoStepNVERigidGPU(m_sysdef, group_rigid));
    addIntegrationMethod(integrator);
    setDeltaT(m_deltaT);
    }

void FIREEnergyMinimizerRigidGPU::reset()
    {
    m_converged = false;
    m_n_since_negative =  m_nmin+1;
    m_n_since_start = 0;
    m_alpha = m_alpha_start;
    m_was_reset = true;

    shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
    ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(rigid_data->getAngMom(), access_location::device, access_mode::readwrite);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = rigid_data->getNumBodies();
    d_rdata.local_beg = 0;
    d_rdata.local_num = d_rdata.n_bodies;
    d_rdata.vel = vel_handle.data;
    d_rdata.angmom = angmom_handle.data;
    
    exec_conf.gpu[0]->call(bind(gpu_fire_rigid_zero_v, d_rdata));
   
    setDeltaT(m_deltaT_set);
    }

/*! \param timestep is the iteration number
*/
void FIREEnergyMinimizerRigidGPU::update(unsigned int timestep)
    {
    if (m_converged)
        return;
        
    IntegratorTwoStep::update(timestep);
    
    if (timestep % m_nevery != 0)
        return;
        
    unsigned int n_bodies = m_rigid_data->getNumBodies();
    if (n_bodies <= 0)
        {
        cerr << endl << "***Error! FIREENergyMinimizerRigid: There is no rigid body for this integrator" << endl << endl;
        throw runtime_error("Error update for FIREEnergyMinimizerRigid (no rigid body)");
        return;
        }
        
    Scalar Pt(0.0), Pr(0.0);
    Scalar vnorm(0.0), wnorm(0.0);
    Scalar fnorm(0.0), tnorm(0.0);
    Scalar energy(0.0);

    // compute the total energy on the GPU
    // CPU version is Scalar energy = computePotentialEnergy(timesteps) / Scalar(nparticles);
    {
    if (m_prof)
        m_prof->push(exec_conf, "FIRE rigid compute total energy");
    
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadOnlyGPU();
    
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
    ArrayHandle<float> d_partial_sum_pe(m_partial_sum_pe, access_location::device, access_mode::overwrite);
    ArrayHandle<float> d_sum_pe(m_sum_pe, access_location::device, access_mode::overwrite);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();
        
    exec_conf.gpu[0]->call(bind(gpu_fire_compute_sum_pe, 
                                d_pdata[0], 
                                d_index_array.data,
                                group_size,
                                d_net_force.data, 
                                d_sum_pe.data, 
                                d_partial_sum_pe.data, 
                                m_block_size, 
                                m_num_blocks));
    
    m_pdata->release();
    
    if (m_prof)
        m_prof->pop(exec_conf);

    }

    {
    
    ArrayHandle<float> h_sum_pe(m_sum_pe, access_location::host, access_mode::read);
    energy = h_sum_pe.data[0] / Scalar(n_bodies);    
    
    }

    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000)*m_etol;
        }
    
    // sum P, vnorm, fnorm
    {
    if (m_prof)
        m_prof->push(exec_conf, "FIRE rigid P, vnorm, fnorm");
    
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::device, access_mode::read);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = n_bodies;
    d_rdata.local_beg = 0;
    d_rdata.local_num = n_bodies;
    d_rdata.vel = vel_handle.data;
    d_rdata.angvel = angvel_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;

    ArrayHandle<float> d_sum_Pt(m_sum_Pt, access_location::device, access_mode::overwrite);
    ArrayHandle<float> d_sum_Pr(m_sum_Pr, access_location::device, access_mode::overwrite);
    exec_conf.gpu[0]->call(bind(gpu_fire_rigid_compute_sum_all, 
                                    d_rdata, 
                                    d_sum_Pt.data, 
                                    d_sum_Pr.data));
    
    if (m_prof)
        m_prof->pop(exec_conf);            
    }
    
    {
    
    ArrayHandle<float> h_sum_Pt(m_sum_Pt, access_location::host, access_mode::read);
    ArrayHandle<float> h_sum_Pr(m_sum_Pr, access_location::host, access_mode::read);
    
    Pt = h_sum_Pt.data[0];
    vnorm = sqrt(h_sum_Pt.data[1]);
    fnorm = sqrt(h_sum_Pt.data[2]);
    
    Pr = h_sum_Pr.data[0];
    wnorm = sqrt(h_sum_Pr.data[1]);
    tnorm = sqrt(h_sum_Pr.data[2]);
    
    }
    
    printf("f = %g (%g); e = %g (%g)\n", fnorm/sqrt(m_sysdef->getNDimensions() * n_bodies), m_ftol, fabs(energy-m_old_energy), m_etol);

    // Check if convergent
    if ((fnorm/sqrt(m_sysdef->getNDimensions() * n_bodies) < m_ftol || fabs(energy-m_old_energy) < m_etol) && m_n_since_start >= m_run_minsteps)

        {
        m_converged = true;
        return;
        }

    // Update velocities
    {
    if (m_prof)
        m_prof->push(exec_conf, "FIRE rigid update velocities and angular momenta");

    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::device, access_mode::read);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = n_bodies;
    d_rdata.local_beg = 0;
    d_rdata.local_num = n_bodies;
    d_rdata.vel = vel_handle.data;
    d_rdata.angmom = angmom_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;
    
    // Scales velocities and angular momenta
    Scalar invfnorm, invtnorm;
    if (fabs(fnorm) > EPSILON)
        invfnorm = 1.0 / fnorm; 
    else
        invfnorm = 1.0;
        
    if (fabs(tnorm) > EPSILON)    
        invtnorm = 1.0 / tnorm;
    else 
        invtnorm = 1.0; 

    exec_conf.gpu[0]->call(bind(gpu_fire_rigid_update_v, 
                                            d_rdata, 
                                            m_alpha, 
                                            vnorm, 
                                            invfnorm, 
                                            wnorm, 
                                            invtnorm));
    
    
    if (m_prof)
        m_prof->pop(exec_conf);                
    }
     
    Scalar P = Pt + Pr;
    if (P > Scalar(0.0))
        {
        m_n_since_negative++;
        if (m_n_since_negative > m_nmin)
            {
            IntegratorTwoStep::setDeltaT(std::min(m_deltaT * m_finc, m_deltaT_max));
            m_alpha *= m_falpha;
            }
        }
    else if (P <= Scalar(0.0))
        {
        IntegratorTwoStep::setDeltaT(m_deltaT * m_fdec);
        m_alpha = m_alpha_start;
        m_n_since_negative = 0;
        if (m_prof)
            m_prof->push(exec_conf, "FIRE rigid zero velocities");
        
        ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::device, access_mode::readwrite);

        gpu_rigid_data_arrays d_rdata;
        d_rdata.n_bodies = n_bodies;
        d_rdata.local_beg = 0;
        d_rdata.local_num = n_bodies;
        d_rdata.vel = vel_handle.data;
        d_rdata.angmom = angmom_handle.data;
        
        exec_conf.gpu[0]->call(bind(gpu_fire_rigid_zero_v, d_rdata));
        
        if (m_prof)
            m_prof->pop(exec_conf);        
        }
    m_n_since_start++;            
    m_old_energy = energy;
    }


void export_FIREEnergyMinimizerRigidGPU()
    {
    class_<FIREEnergyMinimizerRigidGPU, boost::shared_ptr<FIREEnergyMinimizerRigidGPU>, bases<FIREEnergyMinimizer>, boost::noncopyable>
        ("FIREEnergyMinimizerRigidGPU", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

