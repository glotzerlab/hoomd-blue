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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "QuaternionMath.h"
#include "TwoStepNPHRigid.h"
#include <math.h>
 
/*! \file TwoStepNPHRigid.cc
    \brief Contains code for the TwoStepNPHRigid class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param thermo_all ComputeThermo to compute the pressure of the entire system
    \param tauP NPH pressure period
    \param P Pressure set point
    \param skip_restart Flag indicating if restart info is skipped
*/
TwoStepNPHRigid::TwoStepNPHRigid(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo_group,
                       boost::shared_ptr<ComputeThermo> thermo_all,
                       Scalar tauP,
                       boost::shared_ptr<Variant> P,
                       bool skip_restart)
    : TwoStepNVERigid(sysdef, group, true)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNPHRigid" << endl;

    m_thermo_group = thermo_group;
    m_thermo_all = thermo_all;
    m_partial_scale = false;
    m_pressure = P;

    p_stat = true;

    if (tauP <= 0.0)
        m_exec_conf->msg->warning() << "integrate.nph_rigid: tauP set less than or equal to 0.0" << endl;
    
    p_freq = 1.0 / tauP;
    
    boltz = 1.0;
    chain = 5;
    order = 3;
    iter = 5;

    // allocate memory for thermostat chains

    q_b = new Scalar [chain];
    eta_b = new Scalar [chain];
    eta_dot_b = new Scalar [chain];
    f_eta_b = new Scalar [chain];

    eta_b[0] = eta_dot_b[0] = f_eta_b[0] = 0.0;
    for (unsigned int i = 1; i < chain; i++)
        eta_b[i] = eta_dot_b[i] = f_eta_b[i] = 0.0;

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

TwoStepNPHRigid::~TwoStepNPHRigid()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPHRigid" << endl;

    delete [] q_b;
    delete [] eta_b;
    delete [] eta_dot_b;
    delete [] f_eta_b;
    delete [] w;
    delete [] wdti1;
    delete [] wdti2;
    delete [] wdti4;
    }

/* Set integrator variables for restart info
*/

void TwoStepNPHRigid::setRestartIntegratorVariables()
    {
    // set initial state
    IntegratorVariables v = getIntegratorVariables();

    if (!restartInfoTestValid(v, "nph_rigid", 3))   // since NVT derives from NVE, this is true
        {
        // reset the integrator variable
        v.type = "nph_rigid";
        v.variable.resize(3);
        v.variable[0] = Scalar(0.0);
        v.variable[1] = Scalar(0.0);
        v.variable[2] = Scalar(0.0);
        
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);
    }

/*! 
*/
void TwoStepNPHRigid::setup()
    {
    TwoStepNVERigid::setup();
    
    // retrieve integrator variables from restart files
    IntegratorVariables v = getIntegratorVariables();
    eta_b[0] = v.variable[0];
    eta_dot_b[0] = v.variable[1];
    f_eta_b[0] = v.variable[2];
    
    m_thermo_all->compute(0);
    Scalar temperature = 1.0;

    Scalar p_target = m_pressure->getValue(0);
    m_curr_P = m_thermo_all->getPressure();
    // if it is not valid, assume that the current pressure is the set pressure (this should only happen in very 
    // rare circumstances, usually at the start of the simulation before things are initialize)
    if (isnan(m_curr_P))
        m_curr_P = m_pressure->getValue(0);

    // initialize thermostat chain positions, velocites, forces
    Scalar kt = boltz * temperature;
    Scalar p_mass = kt / (p_freq * p_freq);
    q_b[0] = dimension * dimension * p_mass;
    for (unsigned int i = 1; i < chain; i++)
        {
        q_b[i] = p_mass;
        f_eta_b[i] = (q_b[i] * eta_dot_b[i-1] * eta_dot_b[i-1] - kt)/q_b[i];
        }

    // initialize barostat parameters

    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getL();

    Scalar vol;   // volume
    if (dimension == 2) 
        vol = L.x * L.y;
    else 
        vol = L.x * L.y * L.z;

    // calculate group current temperature

    Scalar akin_t = 0.0f, akin_r = 0.0f;
    ArrayHandle<Scalar> body_mass_handle(m_rigid_data->getBodyMass(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::read);

    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
        akin_t += body_mass_handle.data[body] * (vel_handle.data[body].x * vel_handle.data[body].x +
                                                 vel_handle.data[body].y * vel_handle.data[body].y +  
                                                 vel_handle.data[body].z * vel_handle.data[body].z); 
        akin_r += angmom_handle.data[body].x * angvel_handle.data[body].x
                  + angmom_handle.data[body].y * angvel_handle.data[body].y
                  + angmom_handle.data[body].z * angvel_handle.data[body].z;
        }


    m_curr_group_T = (akin_t + akin_r) / (nf_t + nf_r);
    W = (nf_t + nf_r + dimension) * kt / (p_freq * p_freq);
    epsilon = log(vol) / dimension;

    f_epsilon = dimension * (vol * (m_curr_P - p_target) + m_curr_group_T);
    f_epsilon /= W;

    // update order/timestep-dependent coefficients

    for (unsigned int i = 0; i < order; i++)
        {
        wdti1[i] = w[i] * m_deltaT / iter;
        wdti2[i] = wdti1[i] / 2.0;
        wdti4[i] = wdti1[i] / 4.0;
        }        

    // computes the total number of degrees of freedom used for system temperature compute
    ArrayHandle< unsigned int > h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

    unsigned int non_rigid_count = 0;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        if (h_body.data[i] == NO_BODY) non_rigid_count++;

    unsigned int rigid_dof = m_sysdef->getRigidData()->getNumDOF();
    m_dof = dimension * non_rigid_count + rigid_dof; 
        
    }
    
/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Nose-Hoover
     thermostat and Anderson barostat
*/
void TwoStepNPHRigid::integrateStepOne(unsigned int timestep)
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
        m_prof->push("NPH rigid step 1");
    
    // get box
    BoxDim box = m_pdata->getBox();
    
    Scalar tmp, akin_t, akin_r, scale, scale_t, scale_r, scale_v;
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
            
    // update barostat variables a half step

    tmp = -1.0 * dt_half * eta_dot_b[0];
    scale = exp(tmp);
    epsilon_dot += dt_half * f_epsilon;
    epsilon_dot *= scale;
    epsilon += m_deltaT * epsilon_dot;
    dilation = exp(m_deltaT * epsilon_dot);
    
    // update thermostat coupled to barostat

    update_nhcb(timestep);

    // compute scale variables

    tmp = -1.0 * dt_half * onednft * epsilon_dot;
    scale_t = exp(tmp);
    tmp = -1.0 * dt_half * onednfr * epsilon_dot;
    scale_r = exp(tmp);
    tmp = dt_half * epsilon_dot;
    scale_v = m_deltaT * exp(tmp) * maclaurin_series(tmp);

    akin_t = akin_r = 0.0;

    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
            
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
            
        // step 1.2 - update xcm by full step
        com_handle.data[body].x += scale_v * vel_handle.data[body].x;
        com_handle.data[body].y += scale_v * vel_handle.data[body].y;
        com_handle.data[body].z += scale_v * vel_handle.data[body].z;
        
        box.wrap(com_handle.data[body], body_image_handle.data[body]);
            
        // step 1.3 - apply torque (body coords) to quaternion momentum

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
    
    // remap coordinates and box using dilation

    remap();
    
    if (m_prof)
        m_prof->pop();

    }
        
/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNPHRigid::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;
        
    // compute net forces and torques on rigid bodies from particle forces
    computeForceAndTorque(timestep);
    
    if (m_prof)
        m_prof->push("NPH rigid step 2");

    // get box
    BoxDim box = m_pdata->getBox();
    Scalar3 L = box.getL();

    Scalar tmp, scale_t, scale_r, akin_t, akin_r;
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

    tmp = -1.0 * dt_half * (onednft * epsilon_dot);
    scale_t = exp(tmp);
    tmp = -1.0 * dt_half * (onednfr * epsilon_dot);
    scale_r = exp(tmp);

    akin_t = akin_r = 0.0;
    
    // 2nd step: final integration
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
            
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
        
        akin_r += angmom_handle.data[body].x * angvel_handle.data[body].x
                  + angmom_handle.data[body].y * angvel_handle.data[body].y
                  + angmom_handle.data[body].z * angvel_handle.data[body].z;
        }
    }

    // update barostat    

    Scalar vol;   // volume
    if (dimension == 2) 
        vol = L.x * L.y;
    else 
        vol = L.x * L.y * L.z;

    // compute the current thermodynamic properties
    // m_thermo_group->compute(timestep);
    m_thermo_all->compute(timestep);
        
    // compute pressure for the next half time step
    m_curr_P = m_thermo_all->getPressure();
        
    Scalar p_target = m_pressure->getValue(timestep);

    // compute temperature for the next half time step; 
    m_curr_group_T = (akin_t + akin_r) / (nf_t + nf_r);
    Scalar kt = boltz * 1.0;
    W = (nf_t + nf_r + dimension) * kt / (p_freq * p_freq);
    f_epsilon = dimension * (vol * (m_curr_P - p_target) + m_curr_group_T);
    f_epsilon /= W;
    tmp = exp(-1.0 * dt_half * eta_dot_b[0]);
    epsilon_dot = tmp * epsilon_dot + dt_half * f_epsilon;

    if (m_prof)
        m_prof->pop();

    }
  
void export_TwoStepNPHRigid()
    {
    class_<TwoStepNPHRigid, boost::shared_ptr<TwoStepNPHRigid>, bases<TwoStepNVERigid>, boost::noncopyable>
        ("TwoStepNPHRigid", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       boost::shared_ptr<ComputeThermo>,
                       boost::shared_ptr<ComputeThermo>,
                       Scalar,
                       boost::shared_ptr<Variant> >());
    }

#ifdef WIN32
#pragma warning( pop )
#endif

