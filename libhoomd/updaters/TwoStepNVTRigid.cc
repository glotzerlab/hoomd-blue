/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

/*! \file TwoStepNVTRigid.cc
    \brief Defines the TwoStepNVTRigid class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif


#include <boost/python.hpp>
using namespace boost::python;

#include "QuaternionMath.h"
#include "TwoStepNVTRigid.h"
#include <math.h>

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
 \param group The group of particles this integration method is to work on
 \param thermo compute for thermodynamic quantities
 \param T Controlled temperature
 \param tau Time constant
 \param skip_restart Flag indicating if restart info is skipped
*/
TwoStepNVTRigid::TwoStepNVTRigid(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<ParticleGroup> group,
                                 boost::shared_ptr<ComputeThermo> thermo,
                                 boost::shared_ptr<Variant> T,
                                 Scalar tau,
                                 bool skip_restart) 
: TwoStepNVERigid(sysdef, group, true), m_thermo(thermo), m_temperature(T)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVTRigid" << endl;

    if (tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.nvt_rigid: tau set less than or equal to 0.0" << endl;
    
    t_freq = 1.0 / tau;
    
    boltz = 1.0;
    chain = 5;
    order = 3;
    iter = 5;
    
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
    
    {
    ArrayHandle<Scalar> eta_t_handle(eta_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_r_handle(eta_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> f_eta_t_handle(f_eta_t, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> f_eta_r_handle(f_eta_r, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> w_handle(w, access_location::host, access_mode::readwrite);
        
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
    
    
    eta_t_handle.data[0] = eta_r_handle.data[0] = 0.0;
    eta_dot_t_handle.data[0] = eta_dot_r_handle.data[0] = 0.0;
    f_eta_t_handle.data[0] = f_eta_r_handle.data[0] = 0.0;
    for (unsigned int i = 1; i < chain; i++)
        {
        eta_t_handle.data[i] = eta_r_handle.data[i] = 0.0;
        eta_dot_t_handle.data[i] = eta_dot_r_handle.data[i] = 0.0;
        f_eta_t_handle.data[i] = f_eta_r_handle.data[i] = 0.0;
        }
    }
    
    if (!skip_restart)
        {
        setRestartIntegratorVariables();
        }
    
    }
    
TwoStepNVTRigid::~TwoStepNVTRigid()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVTRigid" << endl;
    }

/* Set integrator variables for restart info
*/

void TwoStepNVTRigid::setRestartIntegratorVariables()
    {
    // set initial state
    IntegratorVariables v = getIntegratorVariables();

    if (!restartInfoTestValid(v, "nvt_rigid", 6))   // since NVT derives from NVE, this is true
        {
        // reset the integrator variable
        v.type = "nvt_rigid";
        v.variable.resize(6);
        v.variable[0] = Scalar(0.0);
        v.variable[1] = Scalar(0.0);
        v.variable[2] = Scalar(0.0);
        v.variable[3] = Scalar(0.0);
        v.variable[4] = Scalar(0.0);
        v.variable[5] = Scalar(0.0);
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);
    }

/* Compute the initial forces/torques
*/

void TwoStepNVTRigid::setup()
    {
    TwoStepNVERigid::setup();
    
    if (m_n_bodies <= 0)
        return;
      
    // initialize thermostats
    // set timesteps, constants
    // store Yoshida-Suzuki integrator parameters
    
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
    
    // retrieve integrator variables from restart files
    IntegratorVariables v = getIntegratorVariables();
    eta_t_handle.data[0] = v.variable[0];
    eta_r_handle.data[0] = v.variable[1];
    eta_dot_r_handle.data[0] = v.variable[2];
    eta_dot_t_handle.data[0] = v.variable[3];
    f_eta_r_handle.data[0] = v.variable[4];
    f_eta_t_handle.data[0] = v.variable[5];
    
    //! Total translational and rotational degrees of freedom of rigid bodies
    nf_t = 3 * m_n_bodies;
    nf_r = 3 * m_n_bodies;
    
    //! Subtract from nf_r one for each singular moment inertia of a rigid body
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
    
        if (fabs(moment_inertia_handle.data[body].x) < EPSILON) nf_r -= 1.0;
        if (fabs(moment_inertia_handle.data[body].y) < EPSILON) nf_r -= 1.0;
        if (fabs(moment_inertia_handle.data[body].z) < EPSILON) nf_r -= 1.0;
        }
        
    Scalar kt = boltz * m_temperature->getValue(0);
    Scalar t_mass = kt / (t_freq * t_freq);
    q_t_handle.data[0] = nf_t * t_mass;
    q_r_handle.data[0] = nf_r * t_mass;
    for (unsigned int i = 1; i < chain; i++)
        q_t_handle.data[i] = q_r_handle.data[i] = t_mass;
        
    // initialize thermostat chain positions, velocites, forces
    
    for (unsigned int i = 1; i < chain; i++)
        {
        f_eta_t_handle.data[i] = q_t_handle.data[i-1] * eta_dot_t_handle.data[i-1] * eta_dot_t_handle.data[i-1] - kt;
        f_eta_r_handle.data[i] = q_r_handle.data[i-1] * eta_dot_r_handle.data[i-1] * eta_dot_r_handle.data[i-1] - kt;
        }
        
    }

/*!
    \param timestep Current time step
*/
void TwoStepNVTRigid::integrateStepOne(unsigned int timestep)
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
        m_prof->push("NVT rigid step 1");
    
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
    Scalar dtfm, dt_half;
    
    dt_half = 0.5 * m_deltaT;
    
    akin_t = akin_r = 0.0;
    
    // rigid data handles
    {
    ArrayHandle<Scalar> body_mass_handle(m_rigid_data->getBodyMass(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::read);
    
    ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> orientation_handle(m_rigid_data->getOrientation(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::readwrite);
    
    ArrayHandle<int3> body_image_handle(m_rigid_data->getBodyImage(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> conjqm_handle(m_rigid_data->getConjqm(), access_location::host, access_mode::readwrite);
    
    ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::read);
    ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::read);
    
    // intialize velocity scale for translation and rotation
    
    tmp = -1.0 * dt_half * eta_dot_t_handle.data[0];
    scale_t = exp(tmp);
    tmp = -1.0 * dt_half * eta_dot_r_handle.data[0];
    scale_r = exp(tmp);
    
    // for each body
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
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
            body_image_handle.data[body].x++;
            }
        else if (com_handle.data[body].x < box.xlo)
            {
            com_handle.data[body].x += Lx;
            body_image_handle.data[body].x--;
            }
            
        if (com_handle.data[body].y >= box.yhi)
            {
            com_handle.data[body].y -= Ly;
            body_image_handle.data[body].y++;
            }
        else if (com_handle.data[body].y < box.ylo)
            {
            com_handle.data[body].y += Ly;
            body_image_handle.data[body].y--;
            }
            
        if (com_handle.data[body].z >= box.zhi)
            {
            com_handle.data[body].z -= Lz;
            body_image_handle.data[body].z++;
            }
        else if (com_handle.data[body].z < box.zlo)
            {
            com_handle.data[body].z += Lz;
            body_image_handle.data[body].z--;
            }
            
        matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], torque_handle.data[body], tbody);
        quatvec(orientation_handle.data[body], tbody, fquat);
        
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
        invquatvec(orientation_handle.data[body], conjqm_handle.data[body], mbody);
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
    }
    
    // update thermostat chain
    update_nhcp(akin_t, akin_r, timestep);    
    
    if (m_prof)
        m_prof->pop();
    
    }

void TwoStepNVTRigid::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;
        
    // compute net forces and torques on rigid bodies from particle forces
    computeForceAndTorque(timestep);
    
    if (m_prof)
        m_prof->push("NVT rigid step 2");
    
    Scalar tmp, scale_t, scale_r;
    Scalar4 mbody, tbody, fquat;
    Scalar dt_half;
    
    dt_half = 0.5 * m_deltaT;
    
    // rigid data handles
    {
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
    ArrayHandle<Scalar4> conjqm_handle(m_rigid_data->getConjqm(), access_location::host, access_mode::readwrite);
    
    ArrayHandle<Scalar> eta_dot_t_handle(eta_dot_t, access_location::host, access_mode::read);
    ArrayHandle<Scalar> eta_dot_r_handle(eta_dot_r, access_location::host, access_mode::read);
    
    // intialize velocity scale for translation and rotation
    
    tmp = -1.0 * dt_half * eta_dot_t_handle.data[0];
    scale_t = exp(tmp);
    tmp = -1.0 * dt_half * eta_dot_r_handle.data[0];
    scale_r = exp(tmp);
    
    
    // 2nd step: final integration
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
        Scalar dtfm = dt_half / body_mass_handle.data[body];
        vel_handle.data[body].x = scale_t * vel_handle.data[body].x + dtfm * force_handle.data[body].x;
        vel_handle.data[body].y = scale_t * vel_handle.data[body].y + dtfm * force_handle.data[body].y;
        vel_handle.data[body].z = scale_t * vel_handle.data[body].z + dtfm * force_handle.data[body].z;
        
        // update conjqm, then transform to angmom, set velocity again
        
        matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], torque_handle.data[body], tbody);
        quatvec(orientation_handle.data[body], tbody, fquat);
        
        conjqm_handle.data[body].x = scale_r * conjqm_handle.data[body].x + m_deltaT * fquat.x;
        conjqm_handle.data[body].y = scale_r * conjqm_handle.data[body].y + m_deltaT * fquat.y;
        conjqm_handle.data[body].z = scale_r * conjqm_handle.data[body].z + m_deltaT * fquat.z;
        conjqm_handle.data[body].w = scale_r * conjqm_handle.data[body].w + m_deltaT * fquat.w;
        
        invquatvec(orientation_handle.data[body], conjqm_handle.data[body], mbody);
        transpose_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], mbody, angmom_handle.data[body]);
        
        angmom_handle.data[body].x *= 0.5;
        angmom_handle.data[body].y *= 0.5;
        angmom_handle.data[body].z *= 0.5;
        
        computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body], ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angvel_handle.data[body]);
        }
    }
    

    if (m_prof)
        m_prof->pop();
    }

void TwoStepNVTRigid::update_nhcp(Scalar akin_t, Scalar akin_r, unsigned int timestep)
    {
    Scalar kt, gfkt_t, gfkt_r, tmp, ms, s, s2;
    Scalar dtv;
    
    dtv = m_deltaT;
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
        
    IntegratorVariables v = getIntegratorVariables();
    v.variable[0] = eta_t_handle.data[0];
    v.variable[1] = eta_r_handle.data[0];
    v.variable[2] = eta_dot_r_handle.data[0];
    v.variable[3] = eta_dot_t_handle.data[0];
    setIntegratorVariables(v);
        
    }

/*! Taylor expansion
    \param x Point to take the expansion

*/
inline Scalar TwoStepNVTRigid::maclaurin_series(Scalar x)
    {
    Scalar x2, x4;
    x2 = x * x;
    x4 = x2 * x2;
    return (1.0 + (1.0/6.0) * x2 + (1.0/120.0) * x4 + (1.0/5040.0) * x2 * x4 + (1.0/362880.0) * x4 * x4);
    }

void export_TwoStepNVTRigid()
    {
    class_<TwoStepNVTRigid, boost::shared_ptr<TwoStepNVTRigid>, bases<TwoStepNVERigid>, boost::noncopyable>
    ("TwoStepNVTRigid", init< boost::shared_ptr<SystemDefinition>, 
    boost::shared_ptr<ParticleGroup>, 
    boost::shared_ptr<ComputeThermo>, 
    boost::shared_ptr<Variant> >())
    .def("setT", &TwoStepNVTRigid::setT)
    .def("setTau", &TwoStepNVTRigid::setTau)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

