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
: TwoStepNVERigid(sysdef, group, true)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVTRigid" << endl;

    m_thermo_group = thermo;
    m_temperature = T;

    t_stat = true;

    if (tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.nvt_rigid: tau set less than or equal to 0.0" << endl;
    
    t_freq = 1.0 / tau;
    
    boltz = 1.0;
    chain = 5;
    order = 3;
    iter = 5;
    
    // allocate memory for thermostat chains
    
    q_t = new Scalar [chain];
    q_r = new Scalar [chain];
    eta_t = new Scalar [chain];
    eta_r = new Scalar [chain];
    eta_dot_t = new Scalar [chain];
    eta_dot_r = new Scalar [chain];
    f_eta_t = new Scalar [chain];
    f_eta_r = new Scalar [chain];

    eta_t[0] = eta_r[0] = 0.0;
    eta_dot_t[0] = eta_dot_r[0] = 0.0;
    f_eta_t[0] = f_eta_r[0] = 0.0;
    for (unsigned int i = 1; i < chain; i++)
        {
        eta_t[i] = eta_r[i] = 0.0;
        eta_dot_t[i] = eta_dot_r[i] = 0.0;
        f_eta_t[i] = f_eta_r[i] = 0.0;
        }
    
    w = new Scalar [order];
    wdti1 = new Scalar [order];
    wdti2 = new Scalar [order];
    wdti4 = new Scalar [order];

    if (order == 3)
        {
        w[0] = 1.0 / (2.0 - pow(2.0, 1.0/3.0));
        w[1] = 1.0 - 2.0*w[0];
        w[2] = w[0];
        }
    else if (order == 5)
        {
        w[0] = 1.0 / (4.0 - pow(4.0, 1.0/3.0));
        w[1] = w[0];
        w[2] = 1.0 - 4.0 * w[0];
        w[3] = w[0];
        w[4] = w[0];
        }

    if (!skip_restart)
        {
        setRestartIntegratorVariables();
        }
    
    }
    
TwoStepNVTRigid::~TwoStepNVTRigid()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVTRigid" << endl;

    delete [] w;
    delete [] wdti1;
    delete [] wdti2;
    delete [] wdti4;
    delete [] q_t;
    delete [] q_r;
    delete [] eta_t;
    delete [] eta_r;
    delete [] eta_dot_t;
    delete [] eta_dot_r;
    delete [] f_eta_t;
    delete [] f_eta_r;
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
    
    // retrieve integrator variables from restart files
    IntegratorVariables v = getIntegratorVariables();
    eta_t[0] = v.variable[0];
    eta_r[0] = v.variable[1];
    eta_dot_r[0] = v.variable[2];
    eta_dot_t[0] = v.variable[3];
    f_eta_r[0] = v.variable[4];
    f_eta_t[0] = v.variable[5];
            
    Scalar kt = boltz * m_temperature->getValue(0);
    Scalar t_mass = kt / (t_freq * t_freq);
    q_t[0] = nf_t * t_mass;
    q_r[0] = nf_r * t_mass;
    for (unsigned int i = 1; i < chain; i++)
        q_t[i] = q_r[i] = t_mass;
        
    // initialize thermostat chain positions, velocites, forces
    
    for (unsigned int i = 1; i < chain; i++)
        {
        f_eta_t[i] = q_t[i-1] * eta_dot_t[i-1] * eta_dot_t[i-1] - kt;
        f_eta_r[i] = q_r[i-1] * eta_dot_r[i-1] * eta_dot_r[i-1] - kt;
        }

    // update order/timestep-dependent coefficients
    
    for (unsigned int i = 0; i < order; i++)
        {
        wdti1[i] = w[i] * m_deltaT / iter;
        wdti2[i] = wdti1[i] / 2.0;
        wdti4[i] = wdti1[i] / 4.0;
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
    
    // intialize velocity scale for translation and rotation
    
    scale_t = exp(-dt_half * eta_dot_t[0]);
    scale_r = exp(-dt_half * eta_dot_r[0]);
    
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
        box.wrap(com_handle.data[body], body_image_handle.data[body]);
        
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
    
    Scalar scale_t, scale_r;
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
    
    // intialize velocity scale for translation and rotation
    
    scale_t = exp(-dt_half * eta_dot_t[0]);
    scale_r = exp(-dt_half * eta_dot_r[0]);
    
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
        
        computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body], 
                               ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angvel_handle.data[body]);
        }
    }
    

    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepNVTRigid()
    {
    class_<TwoStepNVTRigid, boost::shared_ptr<TwoStepNVTRigid>, bases<TwoStepNVERigid>, boost::noncopyable>
        ("TwoStepNVTRigid", init< boost::shared_ptr<SystemDefinition>, 
        boost::shared_ptr<ParticleGroup>, 
        boost::shared_ptr<ComputeThermo>, 
        boost::shared_ptr<Variant> >());
    }

#ifdef WIN32
#pragma warning( pop )
#endif

