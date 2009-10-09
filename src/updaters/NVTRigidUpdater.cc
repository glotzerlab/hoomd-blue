/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

/*! \file NVTRigidUpdater.cc
    \brief Defines the NVTRigidUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif


#include <boost/python.hpp>

using namespace boost::python;

#include "NVTRigidUpdater.h"
#include "SystemDefinition.h"
#include <math.h>

#include <boost/bind.hpp>

using namespace boost::signals;
using namespace boost;

using namespace std;

#define EPSILON 1.0e-7

/*! \param pdata Particle data to update
    \param deltaT Time step to use
*/
NVTRigidUpdater::NVTRigidUpdater(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT, boost::shared_ptr<Variant> temperature) : NVERigidUpdater(sysdef, deltaT)
    {
    m_temperature = temperature;
    
    boltz = 1.0;
    chain = 5;
    order = 3;
    iter = 1;
    t_freq = 10.0;
    }

void NVTRigidUpdater::setup()
    {
    NVERigidUpdater::setup();
    
    // allocate memory for thermostat chains
    
    GPUArray<Scalar> q_t_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> q_r_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_t_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_r_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_dot_t_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_dot_r_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> f_eta_t_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> f_eta_r_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> w_alloc(order, m_pdata->getExecConf());
    GPUArray<Scalar> wdti1_alloc(order, m_pdata->getExecConf());
    GPUArray<Scalar> wdti2_alloc(order, m_pdata->getExecConf());
    GPUArray<Scalar> wdti4_alloc(order, m_pdata->getExecConf());
    GPUArray<Scalar4> conjqm_alloc(m_n_bodies, m_pdata->getExecConf());
    
    q_t.swap(q_t_alloc);
    q_r.swap(q_r_alloc);
    eta_t.swap(eta_t_alloc);
    eta_r.swap(eta_r_alloc);
    eta_dot_t.swap(eta_dot_t_alloc);
    eta_dot_r.swap(eta_dot_r_alloc);
    f_eta_t.swap(f_eta_t_alloc);
    f_eta_r.swap(f_eta_r_alloc);
    w.swap(w_alloc);
    wdti1.swap(wdti1_alloc);
    wdti2.swap(wdti2_alloc);
    wdti4.swap(wdti4_alloc);
    conjqm.swap(conjqm_alloc);
    
    
    // initialize thermostats
    // set timesteps, constants
    // store Yoshida-Suzuki integrator parameters
    
        {
        
        ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> orientation_handle(m_rigid_data->getOrientation(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar> q_t_handle(q_t, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> q_r_handle(q_r, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> eta_t_handle(eta_t, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> eta_r_handle(eta_r, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> f_eta_t_handle(f_eta_t, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> f_eta_r_handle(f_eta_r, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> w_handle(w, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> conjqm_handle(conjqm, access_location::host, access_mode::readwrite);
        
        //! Total translational and rotational degrees of freedom of rigid bodies
        nf_t = 3 * m_n_bodies;
        nf_r = 3 * m_n_bodies;
        
        //! Subtract from nf_r one for each singular moment inertia of a rigid body
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
            if (fabs(moment_inertia_handle.data[body].x) < EPSILON) nf_r -= 1.0;
            if (fabs(moment_inertia_handle.data[body].y) < EPSILON) nf_r -= 1.0;
            if (fabs(moment_inertia_handle.data[body].z) < EPSILON) nf_r -= 1.0;
            }
            
        if (order == 3)
            {
            w_handle.data[0] = 1.0 / (2.0 - pow(2.0, 1.0/3.0));
            w_handle.data[1] = 1.0 - 2.0*w_handle.data[0];
            w_handle.data[2] = w_handle.data[0];
            }
        else if (order == 5)
            {
            w_handle.data[0] = 1.0 / (4.0 - pow(4.0, 1.0/3.0));
            w_handle.data[1] = w_handle.data[0];
            w_handle.data[2] = 1.0 - 4.0 * w_handle.data[0];
            w_handle.data[3] = w_handle.data[0];
            w_handle.data[4] = w_handle.data[0];
            }
            
        Scalar4 mbody;
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
            matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angmom_handle.data[body], mbody);
            quat_multiply(orientation_handle.data[body], mbody, conjqm_handle.data[body]);
            
            conjqm_handle.data[body].x *= 2.0;
            conjqm_handle.data[body].y *= 2.0;
            conjqm_handle.data[body].z *= 2.0;
            conjqm_handle.data[body].w *= 2.0;
            }
            
        Scalar kt = boltz * m_temperature->getValue(0);
        Scalar t_mass = kt / (t_freq * t_freq);
        q_t_handle.data[0] = nf_t * t_mass;
        q_r_handle.data[0] = nf_r * t_mass;
        for (unsigned int i = 1; i < chain; i++)
            q_t_handle.data[i] = q_r_handle.data[i] = t_mass;
            
        // initialize thermostat chain positions, velocites, forces
        
        eta_t_handle.data[0] = eta_r_handle.data[0] = 0.0;
        eta_dot_t_handle.data[0] = eta_dot_r_handle.data[0] = 0.0;
        f_eta_t_handle.data[0] = f_eta_r_handle.data[0] = 0.0;
        for (unsigned int i = 1; i < chain; i++)
            {
            eta_t_handle.data[i] = eta_r_handle.data[i] = 0.0;
            eta_dot_t_handle.data[i] = eta_dot_r_handle.data[i] = 0.0;
            f_eta_t_handle.data[i] = q_t_handle.data[i-1] * eta_dot_t_handle.data[i-1] * eta_dot_t_handle.data[i-1] - kt;
            f_eta_r_handle.data[i] = q_r_handle.data[i-1] * eta_dot_r_handle.data[i-1] * eta_dot_r_handle.data[i-1] - kt;
            }
            
        }
        
    }

/*!

*/
void NVTRigidUpdater::initialIntegrate(unsigned int timestep)
    {
    // get box
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    Scalar tmp, akin_t, akin_r, scale_t, scale_r;
    Scalar4 mbody, tbody, fquat;
    Scalar dtfm, dtq, dtv, dt_half;
    
    dt_half = 0.5 * m_deltaT;
    
    akin_t = akin_r = 0.0;
    
    // now we can get on with the velocity verlet: initial integration
    
        {
        
        // rigid data handles
        ArrayHandle<Scalar> body_mass_handle(m_rigid_data->getBodyMass(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> orientation_handle(m_rigid_data->getOrientation(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::readwrite);
        
        ArrayHandle<int> body_imagex_handle(m_rigid_data->getBodyImagex(), access_location::host, access_mode::readwrite);
        ArrayHandle<int> body_imagey_handle(m_rigid_data->getBodyImagey(), access_location::host, access_mode::readwrite);
        ArrayHandle<int> body_imagez_handle(m_rigid_data->getBodyImagez(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::readwrite);
        
        ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::read);
        ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> conjqm_handle(conjqm, access_location::host, access_mode::readwrite);
        
        // intialize velocity scale for translation and rotation
        
        tmp = -1.0 * dt_half * eta_dot_t_handle.data[0];
        scale_t = exp(tmp);
        tmp = -1.0 * dt_half * eta_dot_r_handle.data[0];
        scale_r = exp(tmp);
        
        // for each body
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
            dtfm = dt_half / body_mass_handle.data[body];
            vel_handle.data[body].x += dtfm * force_handle.data[body].x;
            vel_handle.data[body].y += dtfm * force_handle.data[body].y;
            vel_handle.data[body].z += dtfm * force_handle.data[body].z;
            
            vel_handle.data[body].x *= scale_t;
            vel_handle.data[body].y *= scale_t;
            vel_handle.data[body].z *= scale_t;
            
            tmp = vel_handle.data[body].x * vel_handle.data[body].x + vel_handle.data[body].y * vel_handle.data[body].y +
                  vel_handle.data[body].z * vel_handle.data[body].z;
            akin_t += body_mass_handle.data[body] * tmp;
            
            com_handle.data[body].x += vel_handle.data[body].x * m_deltaT;
            com_handle.data[body].y += vel_handle.data[body].y * m_deltaT;
            com_handle.data[body].z += vel_handle.data[body].z * m_deltaT;
            
            // map the center of mass to the periodic box, update the com image info
            if (com_handle.data[body].x >= box.xhi)
                {
                com_handle.data[body].x -= Lx;
                body_imagex_handle.data[body]--;
                }
            else if (com_handle.data[body].x < box.xlo)
                {
                com_handle.data[body].x += Lx;
                body_imagex_handle.data[body]++;
                }
                
            if (com_handle.data[body].y >= box.yhi)
                {
                com_handle.data[body].y -= Ly;
                body_imagey_handle.data[body]--;
                }
            else if (com_handle.data[body].y < box.ylo)
                {
                com_handle.data[body].y += Ly;
                body_imagey_handle.data[body]++;
                }
                
            if (com_handle.data[body].z >= box.zhi)
                {
                com_handle.data[body].z -= Lz;
                body_imagez_handle.data[body]--;
                }
            else if (com_handle.data[body].z < box.zlo)
                {
                com_handle.data[body].z += Lz;
                body_imagez_handle.data[body]++;
                }
                
            matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], torque_handle.data[body], tbody);
            quat_multiply(orientation_handle.data[body], tbody, fquat);
            
            conjqm_handle.data[body].x += m_deltaT * fquat.x;
            conjqm_handle.data[body].y += m_deltaT * fquat.y;
            conjqm_handle.data[body].z += m_deltaT * fquat.z;
            conjqm_handle.data[body].w += m_deltaT * fquat.w;
            
            conjqm_handle.data[body].x *= scale_r;
            conjqm_handle.data[body].y *= scale_r;
            conjqm_handle.data[body].z *= scale_r;
            conjqm_handle.data[body].w *= scale_r;
            
            // step 1.4 to 1.13 - use no_squish rotate to update p and q
            
            no_squish_rotate(3, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], dt_half);
            no_squish_rotate(2, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], dt_half);
            no_squish_rotate(1, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], m_deltaT);
            no_squish_rotate(2, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], dt_half);
            no_squish_rotate(3, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], dt_half);
            
            // update the exyz_space
            // transform p back to angmom
            // update angular velocity
            
            exyzFromQuaternion(orientation_handle.data[body], ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body]);
            inv_quat_multiply(orientation_handle.data[body], conjqm_handle.data[body], mbody);
            transpose_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], mbody, angmom_handle.data[body]);
            
            angmom_handle.data[body].x *= 0.5;
            angmom_handle.data[body].y *= 0.5;
            angmom_handle.data[body].z *= 0.5;
            
            computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body],
                                   ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angvel_handle.data[body]);
                                   
            akin_r += angmom_handle.data[body].x * angvel_handle.data[body].x
                      + angmom_handle.data[body].y * angvel_handle.data[body].y
                      + angmom_handle.data[body].z * angvel_handle.data[body].z;
                      
            }
            
        } // out of scope for handles
        
    // update thermostat chain
    update_nhcp(akin_t, akin_r, timestep);
    
    
    // set positions and velocities of particles in rigid bodies
    set_xv();
    
    }

void NVTRigidUpdater::finalIntegrate(unsigned int timestep)
    {
    // compute net forces and torques on rigid bodies from particle forces
    computeForceAndTorque();
    
    Scalar tmp, scale_t, scale_r;
    Scalar4 mbody, tbody, fquat;
    Scalar dtq, dtv, dt_half;
    
    dt_half = 0.5 * m_deltaT;
    
        {
        
        // rigid data handes
        ArrayHandle<Scalar> body_mass_handle(m_rigid_data->getBodyMass(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> orientation_handle(m_rigid_data->getOrientation(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::readwrite);
        
        ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::read);
        ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> conjqm_handle(conjqm, access_location::host, access_mode::readwrite);
        
        // intialize velocity scale for translation and rotation
        
        tmp = -1.0 * dt_half * eta_dot_t_handle.data[0];
        scale_t = exp(tmp);
        tmp = -1.0 * dt_half * eta_dot_r_handle.data[0];
        scale_r = exp(tmp);
        
        
        // 2nd step: final integration
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
            Scalar dtfm = dt_half / body_mass_handle.data[body];
            vel_handle.data[body].x = scale_t * vel_handle.data[body].x + dtfm * force_handle.data[body].x;
            vel_handle.data[body].y = scale_t * vel_handle.data[body].y + dtfm * force_handle.data[body].y;
            vel_handle.data[body].z = scale_t * vel_handle.data[body].z + dtfm * force_handle.data[body].z;
            
            // update conjqm, then transform to angmom, set velocity again
            // virial is already setup from initial_integrate
            
            matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], torque_handle.data[body], tbody);
            quat_multiply(orientation_handle.data[body], tbody, fquat);
            
            conjqm_handle.data[body].x = scale_r * conjqm_handle.data[body].x + m_deltaT * fquat.x;
            conjqm_handle.data[body].y = scale_r * conjqm_handle.data[body].y + m_deltaT * fquat.y;
            conjqm_handle.data[body].z = scale_r * conjqm_handle.data[body].z + m_deltaT * fquat.z;
            conjqm_handle.data[body].w = scale_r * conjqm_handle.data[body].w + m_deltaT * fquat.w;
            
            inv_quat_multiply(orientation_handle.data[body], conjqm_handle.data[body], mbody);
            transpose_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], mbody, angmom_handle.data[body]);
            
            angmom_handle.data[body].x *= 0.5;
            angmom_handle.data[body].y *= 0.5;
            angmom_handle.data[body].z *= 0.5;
            
            computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body], ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angvel_handle.data[body]);
            }
        } // out of scope for handles
        
    // set velocities of particles in rigid bodies
    set_v();
    
    }

void NVTRigidUpdater::update_nhcp(Scalar akin_t, Scalar akin_r, unsigned int timestep)
    {
    Scalar kt, gfkt_t, gfkt_r, tmp, ms, s, s2;
    Scalar dtv;
    
    dtv = m_deltaT;
    kt = boltz * m_temperature->getValue(timestep);
    gfkt_t = nf_t * kt;
    gfkt_r = nf_r * kt;
    
        {
        
        ArrayHandle<Scalar> q_t_handle(q_t, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> q_r_handle(q_r, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> eta_t_handle(eta_t, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> eta_r_handle(eta_r, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> f_eta_t_handle(f_eta_t, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> f_eta_r_handle(f_eta_r, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> w_handle(w, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> wdti1_handle(wdti1, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> wdti2_handle(wdti2, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> wdti4_handle(wdti4, access_location::host, access_mode::readwrite);
        
        // update thermostat masses
        
        Scalar t_mass = boltz * m_temperature->getValue(timestep) / (t_freq * t_freq);
        q_t_handle.data[0] = nf_t * t_mass;
        q_r_handle.data[0] = nf_r * t_mass;
        for (unsigned int i = 1; i < chain; i++)
            q_t_handle.data[i] = q_r_handle.data[i] = t_mass;
            
        // update order/timestep-dependent coefficients
        
        for (unsigned int i = 0; i < order; i++)
            {
            wdti1_handle.data[i] = w_handle.data[i] * dtv / iter;
            wdti2_handle.data[i] = wdti1_handle.data[i] / 2.0;
            wdti4_handle.data[i] = wdti1_handle.data[i] / 4.0;
            }
            
        // update force of thermostats coupled to particles
        
        f_eta_t_handle.data[0] = (akin_t - gfkt_t) / q_t_handle.data[0];
        f_eta_r_handle.data[0] = (akin_r - gfkt_r) / q_r_handle.data[0];
        
        // multiple timestep iteration
        
        for (unsigned int i = 0; i < iter; i++)
            {
            for (unsigned int j = 0; j < order; j++)
                {
                
                // update thermostat velocities half step
                
                eta_dot_t_handle.data[chain-1] += wdti2_handle.data[j] * f_eta_t_handle.data[chain-1];
                eta_dot_r_handle.data[chain-1] += wdti2_handle.data[j] * f_eta_r_handle.data[chain-1];
                
                for (unsigned int k = 1; k < chain; k++)
                    {
                    tmp = wdti4_handle.data[j] * eta_dot_t_handle.data[chain-k];
                    ms = maclaurin_series(tmp);
                    s = exp(-1.0 * tmp);
                    s2 = s * s;
                    eta_dot_t_handle.data[chain-k-1] = eta_dot_t_handle.data[chain-k-1] * s2 + wdti2_handle.data[j] * f_eta_t_handle.data[chain-k-1] * s * ms;
                    
                    tmp = wdti4_handle.data[j] * eta_dot_r_handle.data[chain-k];
                    ms = maclaurin_series(tmp);
                    s = exp(-1.0 * tmp);
                    s2 = s * s;
                    eta_dot_r_handle.data[chain-k-1] = eta_dot_r_handle.data[chain-k-1] * s2 + wdti2_handle.data[j] * f_eta_r_handle.data[chain-k-1] * s * ms;
                    }
                    
                // update thermostat positions a full step
                
                for (unsigned int k = 0; k < chain; k++)
                    {
                    eta_t_handle.data[k] += wdti1_handle.data[j] * eta_dot_t_handle.data[k];
                    eta_r_handle.data[k] += wdti1_handle.data[j] * eta_dot_r_handle.data[k];
                    }
                    
                // update thermostat forces
                
                for (unsigned int k = 1; k < chain; k++)
                    {
                    f_eta_t_handle.data[k] = q_t_handle.data[k-1] * eta_dot_t_handle.data[k-1] * eta_dot_t_handle.data[k-1] - kt;
                    f_eta_t_handle.data[k] /= q_t_handle.data[k];
                    f_eta_r_handle.data[k] = q_r_handle.data[k-1] * eta_dot_r_handle.data[k-1] * eta_dot_r_handle.data[k-1] - kt;
                    f_eta_r_handle.data[k] /= q_r_handle.data[k];
                    }
                    
                // update thermostat velocities a full step
                
                for (unsigned int k = 0; k < chain-1; k++)
                    {
                    tmp = wdti4_handle.data[j] * eta_dot_t_handle.data[k+1];
                    ms = maclaurin_series(tmp);
                    s = exp(-1.0 * tmp);
                    s2 = s * s;
                    eta_dot_t_handle.data[k] = eta_dot_t_handle.data[k] * s2 + wdti2_handle.data[j] * f_eta_t_handle.data[k] * s * ms;
                    tmp = q_t_handle.data[k] * eta_dot_t_handle.data[k] * eta_dot_t_handle.data[k] - kt;
                    f_eta_t_handle.data[k+1] = tmp / q_t_handle.data[k+1];
                    
                    tmp = wdti4_handle.data[j] * eta_dot_r_handle.data[k+1];
                    ms = maclaurin_series(tmp);
                    s = exp(-1.0 * tmp);
                    s2 = s * s;
                    eta_dot_r_handle.data[k] = eta_dot_r_handle.data[k] * s2 + wdti2_handle.data[j] * f_eta_r_handle.data[k] * s * ms;
                    tmp = q_r_handle.data[k] * eta_dot_r_handle.data[k] * eta_dot_r_handle.data[k] - kt;
                    f_eta_r_handle.data[k+1] = tmp / q_r_handle.data[k+1];
                    }
                    
                eta_dot_t_handle.data[chain-1] += wdti2_handle.data[j] * f_eta_t_handle.data[chain-1];
                eta_dot_r_handle.data[chain-1] += wdti2_handle.data[j] * f_eta_r_handle.data[chain-1];
                }
            }
            
        } // end of scope for handles
        
    }

// Apply evolution operators to quat, quat momentum (see ref. Miller)

void NVTRigidUpdater::no_squish_rotate(unsigned int k, Scalar4& p, Scalar4& q, Scalar4& inertia, Scalar dt)
    {
    Scalar phi, c_phi, s_phi;
    Scalar4 kp, kq;
    
    // apply permuation operator on p and q, get kp and kq
    if (k == 1)
        {
        kq.x = -q.y;  kp.x = -p.y;
        kq.y =  q.x;  kp.y =  p.x;
        kq.z =  q.w;  kp.z =  p.w;
        kq.w = -q.z;  kp.w = -p.z;
        }
    else if (k == 2)
        {
        kq.x = -q.z;  kp.x = -p.z;
        kq.y = -q.w;  kp.y = -p.w;
        kq.z =  q.x;  kp.z =  p.x;
        kq.w =  q.y;  kp.w =  p.y;
        }
    else if (k == 3)
        {
        kq.x = -q.w;  kp.x = -p.w;
        kq.y =  q.z;  kp.y =  p.z;
        kq.z = -q.y;  kp.z = -p.y;
        kq.w =  q.x;  kp.w =  p.x;
        }
        
    // obtain phi, cosines and sines
    
    phi = p.x * kq.x + p.y * kq.y + p.z * kq.z + p.w * kq.w;
    
    Scalar inertia_t;
    if (k == 1) inertia_t = inertia.x;
    else if (k == 2) inertia_t = inertia.y;
    else if (k == 3) inertia_t = inertia.z;
    else inertia_t = Scalar(0.0);
    if (fabs(inertia_t) < EPSILON) phi *= 0.0;
    else phi /= 4.0 * inertia_t;
    
    c_phi = cos(dt * phi);
    s_phi = sin(dt * phi);
    
    // advance p and q
    
    p.x = c_phi * p.x + s_phi * kp.x;
    p.y = c_phi * p.y + s_phi * kp.y;
    p.z = c_phi * p.z + s_phi * kp.z;
    p.w = c_phi * p.w + s_phi * kp.w;
    
    q.x = c_phi * q.x + s_phi * kq.x;
    q.y = c_phi * q.y + s_phi * kq.y;
    q.z = c_phi * q.z + s_phi * kq.z;
    q.w = c_phi * q.w + s_phi * kq.w;
    }

// Quaternion multiply: c = a*b where a is a quaternion, b = (0, b)

void NVTRigidUpdater::quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c)
    {
    c.x = -a.y * b.x - a.z * b.y - a.w * b.z;
    c.y =  a.x * b.x - a.w * b.y + a.z * b.z;
    c.z =  a.w * b.x + a.x * b.y - a.y * b.z;
    c.w = -a.z * b.x + a.y * b.y + a.x * b.z;
    }

// Quaternion multiply: c = inv(a)*b where a is a quaternion, b is a four component vector and c is a three component vector.

void NVTRigidUpdater::inv_quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c)
    {
    c.x = -a.y * b.x + a.x * b.y + a.w * b.z - a.z * b.w;
    c.y = -a.z * b.x - a.w * b.y + a.x * b.z + a.y * b.w;
    c.z = -a.w * b.x + a.z * b.y - a.y * b.z + a.x * b.w;
    }

// Matrix dot: c = dot(A, b) where rows of A are ax, ay, az

void NVTRigidUpdater::matrix_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c)
    {
    c.x = ax.x * b.x + ax.y * b.y + ax.z * b.z;
    c.y = ay.x * b.x + ay.y * b.y + ay.z * b.z;
    c.z = az.x * b.x + az.y * b.y + az.z * b.z;
    }

// Matrix transpose dot: c = dot(trans(A), b) where rows of A are ax, ay, az

void NVTRigidUpdater::transpose_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c)
    {
    c.x = ax.x * b.x + ay.x * b.y + az.x * b.z;
    c.y = ax.y * b.x + ay.y * b.y + az.y * b.z;
    c.z = ax.z * b.x + ay.z * b.y + az.z * b.z;
    }

inline Scalar NVTRigidUpdater::maclaurin_series(Scalar x)
    {
    Scalar x2, x4;
    x2 = x * x;
    x4 = x2 * x2;
    return (1.0 + (1.0/6.0) * x2 + (1.0/120.0) * x4 + (1.0/5040.0) * x2 * x4 + (1.0/362880.0) * x4 * x4);
    }


#ifdef WIN32
#pragma warning( pop )
#endif

