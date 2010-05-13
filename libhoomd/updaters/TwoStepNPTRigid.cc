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

// $Id: TwoStepNPTRigid.cc 2674 2010-01-28 15:16:21Z ndtrung $
// $URL: http://codeblue.umich.edu/hoomd-blue/svn/branches/rigid-bodies/libhoomd/updaters/TwoStepNPTRigid.cc $
// Maintainer: ndtrung

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "TwoStepNPTRigid.h"
#include <math.h>

//! Tolerance for zero principle moments
#define EPSILON 1.0e-7
 
/*! \file TwoStepNPTRigid.cc
    \brief Contains code for the TwoStepNPTRigid class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param thermo_all ComputeThermo to compute the pressure of the entire system
    \param tau NPT temperature period
    \param tauP NPT pressure period
    \param T Temperature set point
    \param P Pressure set point
*/
TwoStepNPTRigid::TwoStepNPTRigid(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo_group,
                       boost::shared_ptr<ComputeThermo> thermo_all,
                       Scalar tau,
                       Scalar tauP,
                       boost::shared_ptr<Variant> T,
                       boost::shared_ptr<Variant> P)
    : TwoStepNVERigid(sysdef, group), m_thermo_group(thermo_group), m_thermo_all(thermo_all), m_partial_scale(false), m_temperature(T), m_pressure(P)
    {
    if (tau <= 0.0)
        cout << "***Warning! tau set less than or equal 0.0 in TwoStepNPTRigid" << endl;
    if (tauP <= 0.0)
        cout << "***Warning! tauP set less than or equal to 0.0 in TwoStepNPTRigid" << endl;
    
    t_freq = 1.0 / tau;
    p_freq = 1.0 / tauP;
    
    boltz = 1.0;
    chain = 5;
    }

/*! 
*/
void TwoStepNPTRigid::setup()
    {
    TwoStepNVERigid::setup();
    
    // allocate memory for thermostat chains
    
    GPUArray<Scalar> q_t_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> q_r_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> q_b_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_t_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_r_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_b_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_dot_t_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_dot_r_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> eta_dot_b_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> f_eta_t_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> f_eta_r_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar> f_eta_b_alloc(chain, m_pdata->getExecConf());
    GPUArray<Scalar4> conjqm_alloc(m_n_bodies, m_pdata->getExecConf());
    
    q_t.swap(q_t_alloc);
    q_r.swap(q_r_alloc);
    q_b.swap(q_b_alloc);
    eta_t.swap(eta_t_alloc);
    eta_r.swap(eta_r_alloc);
    eta_b.swap(eta_b_alloc);
    eta_dot_t.swap(eta_dot_t_alloc);
    eta_dot_r.swap(eta_dot_r_alloc);
    eta_dot_b.swap(eta_dot_b_alloc);
    f_eta_t.swap(f_eta_t_alloc);
    f_eta_r.swap(f_eta_r_alloc);
    f_eta_b.swap(f_eta_b_alloc);
    conjqm.swap(conjqm_alloc);
    
    ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> orientation_handle(m_rigid_data->getOrientation(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
    
    ArrayHandle<Scalar> q_t_handle(q_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> q_r_handle(q_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> q_b_handle(q_b, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_t_handle(eta_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_r_handle(eta_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_b_handle(eta_b, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_dot_b_handle(eta_dot_b, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> f_eta_t_handle(f_eta_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> f_eta_r_handle(f_eta_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> f_eta_b_handle(f_eta_b, access_location::host, access_mode::readwrite);
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
    Scalar p_mass = kt / (p_freq * p_freq);
    dimension = m_sysdef->getNDimensions();
    q_t_handle.data[0] = nf_t * t_mass;
    q_r_handle.data[0] = nf_r * t_mass;
    q_b_handle.data[0] = dimension * dimension * p_mass;
    for (unsigned int k = 1; k < chain; k++)
        {
        q_t_handle.data[k] = q_r_handle.data[k] = t_mass;
        q_b_handle.data[k] = p_mass;
        }    
    
    // initialize thermostat chain positions, velocites, forces
    
    eta_t_handle.data[0] = eta_r_handle.data[0] = eta_b_handle.data[0] = 0.0;
    eta_dot_t_handle.data[0] = eta_dot_r_handle.data[0] = eta_dot_b_handle.data[0] = 0.0;
    f_eta_t_handle.data[0] = f_eta_r_handle.data[0] = f_eta_b_handle.data[0] = 0.0;
    for (unsigned int i = 1; i < chain; i++)
        {
        eta_t_handle.data[i] = eta_r_handle.data[i] = eta_b_handle.data[i] = 0.0;
        eta_dot_t_handle.data[i] = eta_dot_r_handle.data[i] = eta_dot_b_handle.data[i] = 0.0;
        f_eta_t_handle.data[i] = q_t_handle.data[i-1] * eta_dot_t_handle.data[i-1] * eta_dot_t_handle.data[i-1] - kt;
        f_eta_r_handle.data[i] = q_r_handle.data[i-1] * eta_dot_r_handle.data[i-1] * eta_dot_r_handle.data[i-1] - kt;
        f_eta_b_handle.data[i] = q_b_handle.data[i] * eta_dot_b_handle.data[i-1] * eta_dot_b_handle.data[i-1] - kt;
        }
            
    // initialize barostat parameters
    
    const BoxDim& box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    Scalar vol;   // volume
    if (dimension == 2) 
        vol = Lx * Ly;
    else 
        vol = Lx * Ly * Lz;

    w = (nf_t + nf_r + dimension) * kt / (p_freq * p_freq);
    epsilon = log(vol) / dimension;
    epsilon_dot = f_epsilon = 0.0;        

    // computes the total number of degrees of freedom used for system temperature compute
    const ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    unsigned int non_rigid_count = 0;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        if (arrays.body[i] == NO_BODY) non_rigid_count++;

    unsigned int rigid_dof = m_sysdef->getRigidData()->getNumDOF();
    m_dof = dimension * non_rigid_count + rigid_dof; 
    
    // setup virial
    GPUArray<Scalar> viriral_rigid_alloc(m_pdata->getN(), m_pdata->getExecConf());
    virial_rigid.swap(viriral_rigid_alloc);
    
    m_pdata->release();
    }
    
/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Nose-Hoover
     thermostat and Anderson barostat
*/
void TwoStepNPTRigid::integrateStepOne(unsigned int timestep)
    {
    if (m_first_step)
        {
        setup();
        m_first_step = false;
        }
    
    // sanity check
    if (m_n_bodies <= 0)
        return;
        
    if (m_prof)
        m_prof->push("NPT rigid step 1");
    
    // get box
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    Scalar tmp, akin_t, akin_r, scale, scale_t, scale_r, scale_v;
    Scalar4 mbody, tbody, fquat;
    Scalar dtfm, dt_half;
    Scalar onednft, onednfr;
    
    dt_half = 0.5 * m_deltaT;
    
    akin_t = akin_r = 0.0;

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
    
    ArrayHandle<Scalar4> conjqm_handle(conjqm, access_location::host, access_mode::readwrite);
    
    // refresh the virial
    {
    ArrayHandle<Scalar> virial_rigid_handle(virial_rigid, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i<m_pdata->getN(); i++)
        virial_rigid_handle.data[i] = 0.0;
    }
            
    // update barostat variables a half step
    {
    ArrayHandle<Scalar> eta_dot_b_handle(eta_dot_b, access_location::host, access_mode::read);
    
    Scalar kt = boltz * m_temperature->getValue(timestep);
    w = (nf_t + nf_r + dimension) * kt / (p_freq * p_freq);

    tmp = -1.0 * dt_half * eta_dot_b_handle.data[0];
    scale = exp(tmp);
    epsilon_dot += dt_half * f_epsilon;
    epsilon_dot *= scale;
    epsilon += m_deltaT * epsilon_dot;
    dilation = exp(m_deltaT * epsilon_dot);
    }
    
    // update thermostat coupled to barostat

    update_nhcb(timestep);

    // compute scale variables
    {
    ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::read);
    ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::read);
    
    onednft = 1.0 + (Scalar) (dimension) / (double) (nf_t);
    onednfr = (double) (dimension) / (double) (nf_r);

    tmp = -1.0 * dt_half * (eta_dot_t_handle.data[0] + onednft * epsilon_dot);
    scale_t = exp(tmp);
    tmp = -1.0 * dt_half * (eta_dot_r_handle.data[0] + onednfr * epsilon_dot);
    scale_r = exp(tmp);
    tmp = dt_half * epsilon_dot;
    scale_v = m_deltaT * exp(tmp) * maclaurin_series(tmp);
    }
    
    akin_t = akin_r = 0.0;

    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        // step 1.1 - update vcm by 1/2 step
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
        }

    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        // step 1.2 - update xcm by full step
        com_handle.data[body].x += scale_v * vel_handle.data[body].x;
        com_handle.data[body].y += scale_v * vel_handle.data[body].y;
        com_handle.data[body].z += scale_v * vel_handle.data[body].z;
        
        // map the center of mass to the periodic box, update the com image info
        if (com_handle.data[body].x >= box.xhi)
            {
            com_handle.data[body].x -= Lx;
            body_imagex_handle.data[body]++;
            }
        else if (com_handle.data[body].x < box.xlo)
            {
            com_handle.data[body].x += Lx;
            body_imagex_handle.data[body]--;
            }
            
        if (com_handle.data[body].y >= box.yhi)
            {
            com_handle.data[body].y -= Ly;
            body_imagey_handle.data[body]++;
            }
        else if (com_handle.data[body].y < box.ylo)
            {
            com_handle.data[body].y += Ly;
            body_imagey_handle.data[body]--;
            }
            
        if (com_handle.data[body].z >= box.zhi)
            {
            com_handle.data[body].z -= Lz;
            body_imagez_handle.data[body]++;
            }
        else if (com_handle.data[body].z < box.zlo)
            {
            com_handle.data[body].z += Lz;
            body_imagez_handle.data[body]--;
            }
        }

    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        // step 1.3 - apply torque (body coords) to quaternion momentum

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
    
    // remap coordinates and box using dilation

    remap();
    
    // update thermostats

    update_nhcp(akin_t, akin_r, timestep);

    // set positions and velocities of particles in rigid bodies
    set_xv(timestep);
    
    
    if (m_prof)
        m_prof->pop();

    }
        
/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNPTRigid::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;
        
    // compute net forces and torques on rigid bodies from particle forces
    computeForceAndTorque(timestep);
    
    if (m_prof)
        m_prof->push("NPT rigid step 2");
    
    Scalar onednft, onednfr;
    Scalar tmp, scale_t, scale_r, akin_t, akin_r;
    Scalar4 mbody, tbody, fquat;
    Scalar dt_half;
    
    dt_half = 0.5 * m_deltaT;
    
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
  
    onednft = 1.0 + (Scalar) (dimension) / (Scalar) (nf_t);
    onednfr = (Scalar) (dimension) / (Scalar) (nf_r);

    tmp = -1.0 * dt_half * (eta_dot_t_handle.data[0] + onednft * epsilon_dot);
    scale_t = exp(tmp);
    tmp = -1.0 * dt_half * (eta_dot_r_handle.data[0] + onednfr * epsilon_dot);
    scale_r = exp(tmp);

    akin_t = akin_r = 0.0;
    
    // 2nd step: final integration
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        Scalar dtfm = dt_half / body_mass_handle.data[body];
        vel_handle.data[body].x = scale_t * vel_handle.data[body].x + dtfm * force_handle.data[body].x;
        vel_handle.data[body].y = scale_t * vel_handle.data[body].y + dtfm * force_handle.data[body].y;
        vel_handle.data[body].z = scale_t * vel_handle.data[body].z + dtfm * force_handle.data[body].z;
        
        tmp = vel_handle.data[body].x * vel_handle.data[body].x + vel_handle.data[body].y * vel_handle.data[body].y +
          vel_handle.data[body].z * vel_handle.data[body].z;
        akin_t += body_mass_handle.data[body] * tmp; 
    
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
        
        computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body], 
                ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angvel_handle.data[body]);
        
        akin_r += angmom_handle.data[body].x * angvel_handle.data[body].x
                  + angmom_handle.data[body].y * angvel_handle.data[body].y
                  + angmom_handle.data[body].z * angvel_handle.data[body].z;
        }
        
    // set velocities of particles in rigid bodies
    set_v(timestep);
    
    if (m_prof)
        m_prof->pop();
   
    // update barostat
    ArrayHandle<Scalar> eta_dot_b_handle(eta_dot_b, access_location::host, access_mode::read);
    const BoxDim& box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    Scalar vol;   // volume
    if (dimension == 2) 
        vol = Lx * Ly;
    else 
        vol = Lx * Ly * Lz;

    // compute the current thermodynamic properties
    m_thermo_group->compute(timestep+1);
    m_thermo_all->compute(timestep+1);
    
    // compute temperature for the next half time step; currently, I'm still using the internal temperature calculation
    m_curr_group_T = (akin_t + akin_r) / (nf_t + nf_r);
    
    // compute pressure for the next half time step
    m_curr_P = m_thermo_all->getPressure();
    
    Scalar p_target = m_pressure->getValue(timestep);
    f_epsilon = dimension * (vol * (m_curr_P - p_target) + m_curr_group_T);
    f_epsilon /= w;
    tmp = exp(-1.0 * dt_half * eta_dot_b_handle.data[0]);
    epsilon_dot = tmp * epsilon_dot + dt_half * f_epsilon;
        
    }

/* Set position and velocity of constituent particles in rigid bodies in the 1st half of integration
   based on the body center of mass and particle relative position in each body frame.
   References for rigid body virial contribution: N. Matyr et al, PRE 1999, 59 (3), 3733.
   \param timestep Current time step
*/
void TwoStepNPTRigid::set_xv(unsigned int timestep)
    {
    // get box
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    Scalar dt_half = 0.5 * m_deltaT;
    
    // handles
    ArrayHandle<unsigned int> body_size_handle(m_rigid_data->getBodySize(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> com(m_rigid_data->getCOM(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
    ArrayHandle<int> body_imagex_handle(m_rigid_data->getBodyImagex(), access_location::host, access_mode::read);
    ArrayHandle<int> body_imagey_handle(m_rigid_data->getBodyImagey(), access_location::host, access_mode::read);
    ArrayHandle<int> body_imagez_handle(m_rigid_data->getBodyImagez(), access_location::host, access_mode::read);
     
    ArrayHandle<unsigned int> particle_indices_handle(m_rigid_data->getParticleIndices(), access_location::host, access_mode::read);
    unsigned int indices_pitch = m_rigid_data->getParticleIndices().getPitch();
    ArrayHandle<Scalar4> particle_pos_handle(m_rigid_data->getParticlePos(), access_location::host, access_mode::read);
    unsigned int particle_pos_pitch = m_rigid_data->getParticlePos().getPitch();
    
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
    Scalar massone;
    Scalar4 fc;
    ArrayHandle<Scalar> virial_rigid_handle(virial_rigid, access_location::host, access_mode::readwrite);
    
    // access the particle data arrays
    ParticleDataArrays arrays = m_pdata->acquireReadWrite();
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    assert(arrays.vx != NULL && arrays.vy != NULL && arrays.vz != NULL);
    assert(arrays.ix != NULL && arrays.iy != NULL && arrays.iz != NULL);
    
    // for each body
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        unsigned int len = body_size_handle.data[body];
        // for each particle
        for (unsigned int j = 0; j < len; j++)
            {
            // get the actual index of particle in the particle arrays
            unsigned int pidx = particle_indices_handle.data[body * indices_pitch + j];
            // get the index of particle in the current rigid body in the particle_pos array
            unsigned int localidx = body * particle_pos_pitch + j;
            
            // project the position in the body frame to the space frame: xr = rotation_matrix * particle_pos
            Scalar xr = ex_space_handle.data[body].x * particle_pos_handle.data[localidx].x
                        + ey_space_handle.data[body].x * particle_pos_handle.data[localidx].y
                        + ez_space_handle.data[body].x * particle_pos_handle.data[localidx].z;
            Scalar yr = ex_space_handle.data[body].y * particle_pos_handle.data[localidx].x
                        + ey_space_handle.data[body].y * particle_pos_handle.data[localidx].y
                        + ez_space_handle.data[body].y * particle_pos_handle.data[localidx].z;
            Scalar zr = ex_space_handle.data[body].z * particle_pos_handle.data[localidx].x
                        + ey_space_handle.data[body].z * particle_pos_handle.data[localidx].y
                        + ez_space_handle.data[body].z * particle_pos_handle.data[localidx].z;
                        
            // x_particle = x_com + xr
            Scalar4 old_pos;
            old_pos.x = arrays.x[pidx];
            old_pos.y = arrays.y[pidx];
            old_pos.z = arrays.z[pidx];
            
            arrays.x[pidx] = com.data[body].x + xr;
            arrays.y[pidx] = com.data[body].y + yr;
            arrays.z[pidx] = com.data[body].z + zr;
            
            // adjust particle images based on body images
            arrays.ix[pidx] = body_imagex_handle.data[body];
            arrays.iy[pidx] = body_imagey_handle.data[body];
            arrays.iz[pidx] = body_imagez_handle.data[body];
            
            if (arrays.x[pidx] >= box.xhi)
                {
                arrays.x[pidx] -= Lx;
                arrays.ix[pidx]++;
                }
            else if (arrays.x[pidx] < box.xlo)
                {
                arrays.x[pidx] += Lx;
                arrays.ix[pidx]--;
                }
                
            if (arrays.y[pidx] >= box.yhi)
                {
                arrays.y[pidx] -= Ly;
                arrays.iy[pidx]++;
                }
            else if (arrays.y[pidx] < box.ylo)
                {
                arrays.y[pidx] += Ly;
                arrays.iy[pidx]--;
                }
                
            if (arrays.z[pidx] >= box.zhi)
                {
                arrays.z[pidx] -= Lz;
                arrays.iz[pidx]++;
                }
            else if (arrays.z[pidx] < box.zlo)
                {
                arrays.z[pidx] += Lz;
                arrays.iz[pidx]--;
                }
            
            Scalar4 old_vel;
            old_vel.x = arrays.vx[pidx];
            old_vel.y = arrays.vy[pidx];
            old_vel.z = arrays.vz[pidx];
            
            // v_particle = v_com + angvel x xr
            arrays.vx[pidx] = vel_handle.data[body].x + angvel_handle.data[body].y * zr - angvel_handle.data[body].z * yr;
            arrays.vy[pidx] = vel_handle.data[body].y + angvel_handle.data[body].z * xr - angvel_handle.data[body].x * zr;
            arrays.vz[pidx] = vel_handle.data[body].z + angvel_handle.data[body].x * yr - angvel_handle.data[body].y * xr;
            
            massone = arrays.mass[pidx];
            fc.x = massone * (arrays.vx[pidx] - old_vel.x) / dt_half - h_net_force.data[pidx].x;
            fc.y = massone * (arrays.vy[pidx] - old_vel.y) / dt_half - h_net_force.data[pidx].y;
            fc.z = massone * (arrays.vz[pidx] - old_vel.z) / dt_half - h_net_force.data[pidx].z; 
            
            virial_rigid_handle.data[pidx] += (0.5 * (old_pos.x * fc.x + old_pos.y * fc.y + old_pos.z * fc.z) / 3.0);
            }
        }
        
    m_pdata->release();
    
    }

/* Set velocity of constituent particles in rigid bodies in the 2nd half of integration
   based on the body center of mass and particle relative position in each body frame.
   References for rigid body virial contribution: N. Matyr et al, PRE 1999, 59 (3), 3733.
   \param timestep Current time step
*/
void TwoStepNPTRigid::set_v(unsigned int timestep)
    {
    // rigid data handles
    ArrayHandle<unsigned int> body_size_handle(m_rigid_data->getBodySize(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
    
    ArrayHandle<unsigned int> particle_indices_handle(m_rigid_data->getParticleIndices(), access_location::host, access_mode::read);
    unsigned int indices_pitch = m_rigid_data->getParticleIndices().getPitch();
    ArrayHandle<Scalar4> particle_pos_handle(m_rigid_data->getParticlePos(), access_location::host, access_mode::read);
    unsigned int particle_pos_pitch = m_rigid_data->getParticlePos().getPitch();
    
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
    Scalar massone;
    Scalar4 fc;
    ArrayHandle<Scalar> virial_rigid_handle(virial_rigid, access_location::host, access_mode::readwrite);
    
    Scalar dt_half = 0.5 * m_deltaT;
    
    // access the particle data arrays
    ParticleDataArrays arrays = m_pdata->acquireReadWrite();
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    assert(arrays.vx != NULL && arrays.vy != NULL && arrays.vz != NULL);
    
    // for each body
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        unsigned int len = body_size_handle.data[body];
        // for each particle
        for (unsigned int j = 0; j < len; j++)
            {
            // get the actual index of particle in the particle arrays
            unsigned int pidx = particle_indices_handle.data[body * indices_pitch + j];
            // get the index of particle in the current rigid body in the particle_pos array
            unsigned int localidx = body * particle_pos_pitch + j;
            
            // project the position in the body frame to the space frame: xr = rotation_matrix * particle_pos
            Scalar xr = ex_space_handle.data[body].x * particle_pos_handle.data[localidx].x
                        + ey_space_handle.data[body].x * particle_pos_handle.data[localidx].y
                        + ez_space_handle.data[body].x * particle_pos_handle.data[localidx].z;
            Scalar yr = ex_space_handle.data[body].y * particle_pos_handle.data[localidx].x
                        + ey_space_handle.data[body].y * particle_pos_handle.data[localidx].y
                        + ez_space_handle.data[body].y * particle_pos_handle.data[localidx].z;
            Scalar zr = ex_space_handle.data[body].z * particle_pos_handle.data[localidx].x
                        + ey_space_handle.data[body].z * particle_pos_handle.data[localidx].y
                        + ez_space_handle.data[body].z * particle_pos_handle.data[localidx].z;
            
            Scalar4 old_pos;
            old_pos.x = arrays.x[pidx];
            old_pos.y = arrays.y[pidx];
            old_pos.z = arrays.z[pidx];
            
            Scalar4 old_vel;
            old_vel.x = arrays.vx[pidx];
            old_vel.y = arrays.vy[pidx];
            old_vel.z = arrays.vz[pidx];
            
            // v_particle = v_com + angvel x xr
            arrays.vx[pidx] = vel_handle.data[body].x + angvel_handle.data[body].y * zr - angvel_handle.data[body].z * yr;
            arrays.vy[pidx] = vel_handle.data[body].y + angvel_handle.data[body].z * xr - angvel_handle.data[body].x * zr;
            arrays.vz[pidx] = vel_handle.data[body].z + angvel_handle.data[body].x * yr - angvel_handle.data[body].y * xr;
            
            massone = arrays.mass[pidx];
            fc.x = massone * (arrays.vx[pidx] - old_vel.x) / dt_half - h_net_force.data[pidx].x;
            fc.y = massone * (arrays.vy[pidx] - old_vel.y) / dt_half - h_net_force.data[pidx].y;
            fc.z = massone * (arrays.vz[pidx] - old_vel.z) / dt_half - h_net_force.data[pidx].z; 

            virial_rigid_handle.data[pidx] += (0.5 * (old_pos.x * fc.x + old_pos.y * fc.y + old_pos.z * fc.z) / 3.0);
            }
        }
        
    m_pdata->release();
    
    }

/*!
    \param timestep Current time step
*/
Scalar TwoStepNPTRigid::computePressure(unsigned int timestep)
    {
    // grab access to the particle data
    const ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    ArrayHandle<Scalar> h_virial(m_pdata->getNetVirial(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> virial_rigid_handle(virial_rigid, access_location::host, access_mode::read);
    
    // sum up the kinetic energy and virial of the whole system
    double ke_total = 0.0;
    double W = 0.0;
    unsigned int nparticles = m_pdata->getN();
    for (unsigned int i=0; i < nparticles; i++)
        {
        ke_total += (double)arrays.mass[i]*((double)arrays.vx[i] * (double)arrays.vx[i] + 
                                            (double)arrays.vy[i] * (double)arrays.vy[i] +
                                            (double)arrays.vz[i] * (double)arrays.vz[i]);
        
        W += h_virial.data[i];
        W += virial_rigid_handle.data[i];
        }
    
    // compute the system temperature for computing pressure
    ke_total *= 0.5;
    Scalar T = Scalar(2.0 * ke_total / m_dof);

    // volume/area & other 2D stuff needed
    BoxDim box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    Scalar vol;
    if (dimension == 2)
        {
        // "volume" is area in 2D
        vol = Lx * Ly;
        // W needs to be corrected since the 1/3 factor is built in
        W *= Scalar(3.0/2.0);
        }
    else
        vol = Lx * Ly * Lz;

    // done!
    m_pdata->release();
    
    // pressure: P = (N * K_B * T + W)/V
    return (m_dof * boltz * T + W) / vol;
    }

/*!
    \param timestep Current time step
*/
Scalar TwoStepNPTRigid::computeTemperature(unsigned int timestep)
    {
    // grab access to the particle data
    const ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    
    // sum up the kinetic energy and virial of the whole system
    double ke_total = 0.0;
    unsigned int nparticles = m_pdata->getN();
    for (unsigned int i=0; i < nparticles; i++)
        {
        ke_total += (double)arrays.mass[i]*((double)arrays.vx[i] * (double)arrays.vx[i] + 
                                            (double)arrays.vy[i] * (double)arrays.vy[i] +
                                            (double)arrays.vz[i] * (double)arrays.vz[i]);
        }

    m_pdata->release();
    
    // compute the system temperature for computing pressure
    ke_total *= 0.5;
    Scalar T = Scalar(2.0 * ke_total / m_dof);   
    
    return T;
    }
    
/*! Calculate the new box size from dilation
    Remap the rigid body COMs from old box to new box
    Note that NPT rigid currently only deals with rigid bodies, no point particles
    For hybrid systems, use TwoStepNPT coupled with TwoStepNVTRigid to avoid duplicating box resize
*/
void TwoStepNPTRigid::remap()
{
    Scalar oldlo, oldhi, ctr;
    Scalar xlo, ylo, zlo, Lx, Ly, Lz, invLx, invLy, invLz;
    
    BoxDim box = m_pdata->getBox();
    
    // convert rigid body COMs to lamda coords

    ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::readwrite);
    
    xlo = box.xlo;
    ylo = box.ylo;
    zlo = box.zlo;
    Lx = box.xhi - box.xlo;
    Ly = box.yhi - box.ylo;
    Lz = box.zhi - box.zlo;
    invLx = 1.0 / Lx;
    invLy = 1.0 / Ly;
    invLz = 1.0 / Lz;
    
    Scalar4 delta;
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        delta.x = com_handle.data[body].x - xlo;
        delta.y = com_handle.data[body].y - ylo;
        delta.z = com_handle.data[body].z - zlo;

        com_handle.data[body].x = invLx * delta.x;
        com_handle.data[body].y = invLy * delta.y;
        com_handle.data[body].z = invLz * delta.z;
        }

    // reset box to new size/shape
    oldlo = box.xlo;
    oldhi = box.xhi;
    ctr = 0.5 * (oldlo + oldhi);
    box.xlo = (oldlo - ctr) * dilation + ctr;
    box.xhi = (oldhi - ctr) * dilation + ctr;
    Lx = box.xhi - box.xlo;
    
    oldlo = box.ylo;
    oldhi = box.yhi;
    ctr = 0.5 * (oldlo + oldhi);
    box.ylo = (oldlo - ctr) * dilation + ctr;
    box.yhi = (oldhi - ctr) * dilation + ctr;
    Ly = box.yhi - box.ylo;
    
    oldlo = box.zlo;
    oldhi = box.zhi;
    ctr = 0.5 * (oldlo + oldhi);
    box.zlo = (oldlo - ctr) * dilation + ctr;
    box.zhi = (oldhi - ctr) * dilation + ctr;
    Lz = box.zhi - box.zlo;
    
    m_pdata->setBox(BoxDim(Lx, Ly, Lz));
    
    // convert rigid body COMs back to box coords
    Scalar4 newboxlo;
    newboxlo.x = -Lx/2.0;
    newboxlo.y = -Ly/2.0;
    newboxlo.z = -Lz/2.0;
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        com_handle.data[body].x = Lx * com_handle.data[body].x + newboxlo.x;
        com_handle.data[body].y = Ly * com_handle.data[body].y + newboxlo.y;
        com_handle.data[body].z = Lz * com_handle.data[body].z + newboxlo.z;
        }
    
    }

/*! Update Nose-Hoover thermostats
    \param akin_t Translational kinetic energy
    \param akin_r Rotational kinetic energy
    \param timestep Current time step
*/
void TwoStepNPTRigid::update_nhcp(Scalar akin_t, Scalar akin_r, unsigned int timestep)
{
    Scalar kt, gfkt_t, gfkt_r, tmp, ms, s, s2;
    Scalar dtv, dtq;
    
    dtv = m_deltaT;
    dtq = 0.5 * m_deltaT;
    kt = boltz * m_temperature->getValue(timestep);
    gfkt_t = nf_t * kt;
    gfkt_r = nf_r * kt;
    
    ArrayHandle<Scalar> q_t_handle(q_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> q_r_handle(q_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_t_handle(eta_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_r_handle(eta_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> f_eta_t_handle(f_eta_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> f_eta_r_handle(f_eta_r, access_location::host, access_mode::readwrite);
        
    // update thermostat masses

    Scalar t_mass = boltz * m_temperature->getValue(timestep) / (t_freq * t_freq);
    q_t_handle.data[0] = nf_t * t_mass;
    q_r_handle.data[0] = nf_r * t_mass;
    for (unsigned int i = 1; i < chain; i++)
        q_t_handle.data[i] = q_r_handle.data[i] = t_mass;
            
    // update force of thermostats coupled to particles
        
    f_eta_t_handle.data[0] = (akin_t - gfkt_t) / q_t_handle.data[0];
    f_eta_r_handle.data[0] = (akin_r - gfkt_r) / q_r_handle.data[0];
        
    // update thermostat velocities half step

    eta_dot_t_handle.data[chain-1] += dtq * f_eta_t_handle.data[chain-1];
    eta_dot_r_handle.data[chain-1] += dtq * f_eta_r_handle.data[chain-1];

    for (unsigned int k = 1; k < chain; k++) 
        {
        tmp = dtq * eta_dot_t_handle.data[chain-k];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_t_handle.data[chain-k-1] = eta_dot_t_handle.data[chain-k-1] * s2 + 
                                dtq * f_eta_t_handle.data[chain-k-1] * s * ms;
          
        tmp = dtq * eta_dot_r_handle.data[chain-k];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_r_handle.data[chain-k-1] = eta_dot_r_handle.data[chain-k-1] * s2 + 
                                dtq * f_eta_r_handle.data[chain-k-1] * s * ms;
        }

    // update thermostat positions a full step

    for (unsigned int k = 0; k < chain; k++) 
        {
        eta_t_handle.data[k] += dtv * eta_dot_t_handle.data[k];
        eta_r_handle.data[k] += dtv * eta_dot_r_handle.data[k];
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
        tmp = dtq * eta_dot_t_handle.data[k+1];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_t_handle.data[k] = eta_dot_t_handle.data[k] * s2 + dtq * f_eta_t_handle.data[k] * s * ms;
        tmp = q_t_handle.data[k] * eta_dot_t_handle.data[k] * eta_dot_t_handle.data[k] - kt;
        f_eta_t_handle.data[k+1] = tmp / q_t_handle.data[k+1];

        tmp = dtq * eta_dot_r_handle.data[k+1];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_r_handle.data[k] = eta_dot_r_handle.data[k] * s2 + dtq * f_eta_r_handle.data[k] * s * ms;
        tmp = q_r_handle.data[k] * eta_dot_r_handle.data[k] * eta_dot_r_handle.data[k] - kt;
        f_eta_r_handle.data[k+1] = tmp / q_r_handle.data[k+1];

        }

    eta_dot_t_handle.data[chain-1] += dtq * f_eta_t_handle.data[chain-1];
    eta_dot_r_handle.data[chain-1] += dtq * f_eta_r_handle.data[chain-1];
    
    }

/*! Update Nose-Hoover barostats
    \param timestep Current time step
*/
void TwoStepNPTRigid::update_nhcb(unsigned int timestep)
    {
    Scalar kt, tmp, ms, s, s2;
    Scalar dt_half;
    
    dt_half = 0.5 * m_deltaT;
    kt = boltz * m_temperature->getValue(timestep);
    
    ArrayHandle<Scalar> q_b_handle(q_b, access_location::host, access_mode::readwrite);    
    ArrayHandle<Scalar> eta_b_handle(eta_b, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_dot_b_handle(eta_dot_b, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> f_eta_b_handle(f_eta_b, access_location::host, access_mode::readwrite);
    
    // update thermostat masses
    
    double tb_mass = kt / (p_freq * p_freq);
    q_b_handle.data[0] = dimension * dimension * tb_mass;
    for (unsigned int i = 1; i < chain; i++) 
        q_b_handle.data[i] = tb_mass;

    // update forces acting on thermostat

    tmp = w * epsilon_dot * epsilon_dot;
    f_eta_b_handle.data[0] = (tmp - kt) / q_b_handle.data[0];

    // update thermostat velocities a half step

    eta_dot_b_handle.data[chain-1] += dt_half * f_eta_b_handle.data[chain-1];

    for (unsigned int k = 1; k < chain; k++) 
        {
        tmp = dt_half * eta_dot_b_handle.data[chain-k];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_b_handle.data[chain-k-1] = eta_dot_b_handle.data[chain-k-1] * s2 + 
                                            dt_half * f_eta_b_handle.data[chain-k-1] * s * ms;
        }

    // update thermostat positions

    for (unsigned int k = 0; k < chain; k++)
        eta_b_handle.data[k] += m_deltaT * eta_dot_b_handle.data[k];

    // update thermostat forces

    for (unsigned int k = 1; k < chain; k++) 
        {
        f_eta_b_handle.data[k] = q_b_handle.data[k-1] * eta_dot_b_handle.data[k-1] * eta_dot_b_handle.data[k-1] - kt;
        f_eta_b_handle.data[k] /= q_b_handle.data[k];
        }

    // update thermostat velocites a full step

    for (unsigned int k = 0; k < chain-1; k++) 
        {
        tmp = dt_half * eta_dot_b_handle.data[k+1];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_b_handle.data[k] = eta_dot_b_handle.data[k] * s2 + dt_half * f_eta_b_handle.data[k] * s * ms;
        tmp = q_b_handle.data[k] * eta_dot_b_handle.data[k] * eta_dot_b_handle.data[k] - kt;
        f_eta_b_handle.data[k+1] = tmp / q_b_handle.data[k+1];
        }

    eta_dot_b_handle.data[chain-1] += dt_half * f_eta_b_handle.data[chain-1];

    }

/*! Apply evolution operators to quat, quat momentum (see ref. Miller)
    \param k Direction
    \param p Thermostat angular momentum conjqm
    \param q Quaternion
    \param inertia Moment of inertia
    \param dt Time step
*/
void TwoStepNPTRigid::no_squish_rotate(unsigned int k, Scalar4& p, Scalar4& q, Scalar4& inertia, Scalar dt)
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
    else
        {
        kq.x = 0.0;  kp.x = 0.0;
        kq.y = 0.0;  kp.y = 0.0;
        kq.z = 0.0;  kp.z = 0.0;
        kq.w = 0.0;  kp.w = 0.0;
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
    normalize(q);
    }

/*! Quaternion multiply: c = a * b 
    \param a Quaternion
    \param b A three component vector
    \param c Returned quaternion
*/
void TwoStepNPTRigid::quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c)
    {
    c.x = -a.y * b.x - a.z * b.y - a.w * b.z;
    c.y =  a.x * b.x - a.w * b.y + a.z * b.z;
    c.z =  a.w * b.x + a.x * b.y - a.y * b.z;
    c.w = -a.z * b.x + a.y * b.y + a.x * b.z;
    }

/*! Inverse quaternion multiply: c = inv(a) * b 
    \param a Quaternion
    \param b A four component vector
    \param c A three component vector
*/
void TwoStepNPTRigid::inv_quat_multiply(Scalar4& a, Scalar4& b, Scalar4& c)
    {
    c.x = -a.y * b.x + a.x * b.y + a.w * b.z - a.z * b.w;
    c.y = -a.z * b.x - a.w * b.y + a.x * b.z + a.y * b.w;
    c.z = -a.w * b.x + a.z * b.y - a.y * b.z + a.x * b.w;
    }

/*! Matrix dot: c = dot(A, b) 
    \param ax The first row of A
    \param ay The second row of A
    \param az The third row of A
    \param b A three component vector
    \param c A three component vector    
*/
void TwoStepNPTRigid::matrix_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c)
    {
    c.x = ax.x * b.x + ax.y * b.y + ax.z * b.z;
    c.y = ay.x * b.x + ay.y * b.y + ay.z * b.z;
    c.z = az.x * b.x + az.y * b.y + az.z * b.z;
    }

/*! Matrix transpose dot: c = dot(trans(A), b) 
    \param ax The first row of A
    \param ay The second row of A
    \param az The third row of A
    \param b A three component vector
    \param c A three component vector
*/
void TwoStepNPTRigid::transpose_dot(Scalar4& ax, Scalar4& ay, Scalar4& az, Scalar4& b, Scalar4& c)
    {
    c.x = ax.x * b.x + ay.x * b.y + az.x * b.z;
    c.y = ax.y * b.x + ay.y * b.y + az.y * b.z;
    c.z = ax.z * b.x + ay.z * b.y + az.z * b.z;
    }

/*! Maclaurine expansion
    \param x Point to take the expansion

*/
inline Scalar TwoStepNPTRigid::maclaurin_series(Scalar x)
    {
    Scalar x2, x4;
    x2 = x * x;
    x4 = x2 * x2;
    return (1.0 + (1.0/6.0) * x2 + (1.0/120.0) * x4 + (1.0/5040.0) * x2 * x4 + (1.0/362880.0) * x4 * x4);
    }

  
void export_TwoStepNPTRigid()
    {
    class_<TwoStepNPTRigid, boost::shared_ptr<TwoStepNPTRigid>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
        ("TwoStepNPTRigid", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       boost::shared_ptr<ComputeThermo>,
                       boost::shared_ptr<ComputeThermo>,
                       Scalar,
                       Scalar,
                       boost::shared_ptr<Variant>,
                       boost::shared_ptr<Variant> >())
        .def("setT", &TwoStepNPTRigid::setT)
        .def("setP", &TwoStepNPTRigid::setP)
        .def("setTau", &TwoStepNPTRigid::setTau)
        .def("setTauP", &TwoStepNPTRigid::setTauP)
        .def("setPartialScale", &TwoStepNPTRigid::setPartialScale)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

