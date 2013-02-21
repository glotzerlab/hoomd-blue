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
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "QuaternionMath.h"
#include "TwoStepNVERigid.h"
#include <math.h>
#include <fstream>

using namespace std;

/*! \file TwoStepNVERigid.cc
 \brief Defines the TwoStepNVERigid class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
 \param group The group of particles this integration method is to work on
 \param skip_restart Skip initialization of the restart information
 */
TwoStepNVERigid::TwoStepNVERigid(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<ParticleGroup> group,
                                 bool skip_restart)
    : IntegrationMethodTwoStep(sysdef, group)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVERigid" << endl;

    if (!skip_restart)
        {
        setRestartIntegratorVariables();
        }
         
    // Get the system rigid data
    m_rigid_data = sysdef->getRigidData();
    
    // Get the particle data associated with the rigid data (i.e. the system particle data?)
    m_pdata = sysdef->getParticleData();
    
    m_first_step = true;
    
    // Create my rigid body group from the particle group
    m_body_group = boost::shared_ptr<RigidBodyGroup>(new RigidBodyGroup(sysdef, m_group));
    if (m_body_group->getNumMembers() == 0)
        {
        m_exec_conf->msg->warning() << "integrate.*_rigid: Empty group." << endl;
        }
    
    w = wdti1 = wdti2 = wdti4 = NULL;
    q_t = q_r = q_b = NULL;
    eta_t = eta_r = eta_b = NULL;
    eta_dot_t = eta_dot_r = eta_dot_b = NULL;
    f_eta_t = f_eta_r = f_eta_b = NULL;
    
    // Using thermostat or barostat 
    t_stat = false;
    p_stat = false;
    }

TwoStepNVERigid::~TwoStepNVERigid()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVERigid" << endl;
    }

void TwoStepNVERigid::setRestartIntegratorVariables()
{
    // set a named, but otherwise blank set of integrator variables
    IntegratorVariables v = getIntegratorVariables();
    
    if (!restartInfoTestValid(v, "nve_rigid", 0))
        {
        v.type = "nve_rigid";
        v.variable.resize(0);
        setValidRestart(false);
        }
    else
        setValidRestart(true);
    
    setIntegratorVariables(v);
}

/* Setup computes the initial body forces and torques prior to the first update step
    
*/

void TwoStepNVERigid::setup()
    {
    if (m_prof)
        m_prof->push("Rigid setup");
        
    // Get the number of rigid bodies for frequent use
    m_n_bodies = m_body_group->getNumMembers();
    
    // Get the system dimensionality    
    dimension = m_sysdef->getNDimensions();
 
    // sanity check
    if (m_n_bodies <= 0)
        return;
        
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GPUArray< Scalar4 >& net_torque = m_pdata->getNetTorqueArray();
    
    {
    // rigid data handles
    ArrayHandle<Scalar> body_mass_handle(m_rigid_data->getBodyMass(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> body_size_handle(m_rigid_data->getBodySize(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> particle_indices_handle(m_rigid_data->getParticleIndices(), access_location::host, access_mode::read);
    unsigned int indices_pitch = m_rigid_data->getParticleIndices().getPitch();
    ArrayHandle<Scalar4> particle_pos_handle(m_rigid_data->getParticlePos(), access_location::host, access_mode::read);
    unsigned int particle_pos_pitch = m_rigid_data->getParticlePos().getPitch();
    
    ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> orientation_handle(m_rigid_data->getOrientation(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> conjqm_handle(m_rigid_data->getConjqm(), access_location::host, access_mode::readwrite);
    
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_torque(net_torque, access_location::host, access_mode::read);
    
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

    g_f = nf_t + nf_r;  
    onednft = 1.0 + (double)(dimension) / (double)g_f;
    onednfr = (double) (dimension) / (double)g_f;

    // Reset all forces and torques
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
        vel_handle.data[body].x = 0.0;
        vel_handle.data[body].y = 0.0;
        vel_handle.data[body].z = 0.0;
        
        force_handle.data[body].x = 0.0;
        force_handle.data[body].y = 0.0;
        force_handle.data[body].z = 0.0;
        
        torque_handle.data[body].x = 0.0;
        torque_handle.data[body].y = 0.0;
        torque_handle.data[body].z = 0.0;
        
        angmom_handle.data[body].x = 0.0;
        angmom_handle.data[body].y = 0.0;
        angmom_handle.data[body].z = 0.0;
        }
        
    // Access the particle data arrays
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);

    // for each body
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
        // for each particle
        unsigned int len = body_size_handle.data[body];
        for (unsigned int j = 0; j < len; j++)
            {
            // get the index of particle in the particle arrays
            unsigned int pidx = particle_indices_handle.data[body * indices_pitch + j];
            
            // get the particle mass
            Scalar mass_one = h_vel.data[pidx].w;
            
            vel_handle.data[body].x += mass_one * h_vel.data[pidx].x;
            vel_handle.data[body].y += mass_one * h_vel.data[pidx].y;
            vel_handle.data[body].z += mass_one * h_vel.data[pidx].z;
            
            Scalar fx, fy, fz;
            fx = h_net_force.data[pidx].x;
            fy = h_net_force.data[pidx].y;
            fz = h_net_force.data[pidx].z;
            
            force_handle.data[body].x += fx;
            force_handle.data[body].y += fy;
            force_handle.data[body].z += fz;
            
            // Torque = r x f (all are in the space frame)
            unsigned int localidx = body * particle_pos_pitch + j;
            Scalar rx = ex_space_handle.data[body].x * particle_pos_handle.data[localidx].x
                    + ey_space_handle.data[body].x * particle_pos_handle.data[localidx].y
                    + ez_space_handle.data[body].x * particle_pos_handle.data[localidx].z;
            Scalar ry = ex_space_handle.data[body].y * particle_pos_handle.data[localidx].x
                    + ey_space_handle.data[body].y * particle_pos_handle.data[localidx].y
                    + ez_space_handle.data[body].y * particle_pos_handle.data[localidx].z;
            Scalar rz = ex_space_handle.data[body].z * particle_pos_handle.data[localidx].x
                    + ey_space_handle.data[body].z * particle_pos_handle.data[localidx].y
                    + ez_space_handle.data[body].z * particle_pos_handle.data[localidx].z;
            
            Scalar tx = h_net_torque.data[pidx].x;
            Scalar ty = h_net_torque.data[pidx].y;
            Scalar tz = h_net_torque.data[pidx].z;
            
            torque_handle.data[body].x += ry * fz - rz * fy + tx;
            torque_handle.data[body].y += rz * fx - rx * fz + ty;
            torque_handle.data[body].z += rx * fy - ry * fx + tz;
            
            // Angular momentum = r x (m * v) is calculated for setup
            angmom_handle.data[body].x += ry * (mass_one * h_vel.data[pidx].z) - rz * (mass_one * h_vel.data[pidx].y);
            angmom_handle.data[body].y += rz * (mass_one * h_vel.data[pidx].x) - rx * (mass_one * h_vel.data[pidx].z);
            angmom_handle.data[body].z += rx * (mass_one * h_vel.data[pidx].y) - ry * (mass_one * h_vel.data[pidx].x);
            }
        
        }
        
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
        vel_handle.data[body].x /= body_mass_handle.data[body];
        vel_handle.data[body].y /= body_mass_handle.data[body];
        vel_handle.data[body].z /= body_mass_handle.data[body];
        
        computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body],
                               ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angvel_handle.data[body]);
        }

    Scalar4 mbody;
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
    
        matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angmom_handle.data[body], mbody);
        quatvec(orientation_handle.data[body], mbody, conjqm_handle.data[body]);
        
        conjqm_handle.data[body].x *= 2.0;
        conjqm_handle.data[body].y *= 2.0;
        conjqm_handle.data[body].z *= 2.0;
        conjqm_handle.data[body].w *= 2.0;
        }
    
    } // out of scope for handles   
    
    
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepNVERigid::integrateStepOne(unsigned int timestep)
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
        m_prof->push("NVE rigid step 1");
    
    // get box
    const BoxDim& box = m_pdata->getBox();
    
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
    
    ArrayHandle<int3> body_image_handle(m_rigid_data->getBodyImage(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::readwrite);
    
    Scalar dt_half = 0.5 * m_deltaT;
    Scalar dtfm;
    
    // for each body
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
        dtfm = dt_half / body_mass_handle.data[body];
        vel_handle.data[body].x += dtfm * force_handle.data[body].x;
        vel_handle.data[body].y += dtfm * force_handle.data[body].y;
        vel_handle.data[body].z += dtfm * force_handle.data[body].z;
        
        com_handle.data[body].x += vel_handle.data[body].x * m_deltaT;
        com_handle.data[body].y += vel_handle.data[body].y * m_deltaT;
        com_handle.data[body].z += vel_handle.data[body].z * m_deltaT;
        
        box.wrap(com_handle.data[body], body_image_handle.data[body]);
        
        // update the angular momentum
        angmom_handle.data[body].x += dt_half * torque_handle.data[body].x;
        angmom_handle.data[body].y += dt_half * torque_handle.data[body].y;
        angmom_handle.data[body].z += dt_half * torque_handle.data[body].z;
        
        // update quaternion and angular velocity
        advanceQuaternion(angmom_handle.data[body],
                          moment_inertia_handle.data[body],
                          angvel_handle.data[body],
                          ex_space_handle.data[body],
                          ey_space_handle.data[body],
                          ez_space_handle.data[body],
                          m_deltaT,
                          orientation_handle.data[body]);
        }
    } // out of scope for handles
        

    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNVERigid::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;
    
    
    // compute net forces and torques on rigid bodies from particle forces
    computeForceAndTorque(timestep);

    if (m_prof)
        m_prof->push("NVE rigid step 2");
    
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
    
    Scalar dt_half = 0.5 * m_deltaT;
    Scalar dtfm;
    
    // 2nd step: final integration
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
        dtfm = dt_half / body_mass_handle.data[body];
        vel_handle.data[body].x += dtfm * force_handle.data[body].x;
        vel_handle.data[body].y += dtfm * force_handle.data[body].y;
        vel_handle.data[body].z += dtfm * force_handle.data[body].z;
        
        angmom_handle.data[body].x += dt_half * torque_handle.data[body].x;
        angmom_handle.data[body].y += dt_half * torque_handle.data[body].y;
        angmom_handle.data[body].z += dt_half * torque_handle.data[body].z;
        
        computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body],
                               ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angvel_handle.data[body]);
        }
    } // out of scope for handles


    if (m_prof)
        m_prof->pop();
    }
    
/*! \param query_group Group over which to count degrees of freedom.
    A majority of the integration methods add D degrees of freedom per particle in \a query_group that is also in the
    group assigned to the method. Hence, the base class IntegrationMethodTwoStep will implement that counting.
    Derived classes can ovveride if needed.
*/
unsigned int TwoStepNVERigid::getNDOF(boost::shared_ptr<ParticleGroup> query_group)
    {
     ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
     
    // count the number of particles both in query_group and m_group
    boost::shared_ptr<ParticleGroup> intersect_particles = ParticleGroup::groupIntersection(m_group, query_group);
    
    RigidBodyGroup intersect_bodies(m_sysdef, intersect_particles);
    
    // Counting body DOF: 
    // 3D systems: a body has 6 DOF by default, subtracted by the number of zero moments of inertia
    // 2D systems: a body has 3 DOF by default
    unsigned int query_group_dof = 0;
    unsigned int dimension = m_sysdef->getNDimensions();
    unsigned int dof_one;
    for (unsigned int group_idx = 0; group_idx < intersect_bodies.getNumMembers(); group_idx++)
        {
        unsigned int body = intersect_bodies.getMemberIndex(group_idx);
        if (m_body_group->isMember(body))
            {
            if (dimension == 3)
                {
                dof_one = 6;
                if (moment_inertia_handle.data[body].x == 0.0)
                    dof_one--;
                
                if (moment_inertia_handle.data[body].y == 0.0)
                    dof_one--;
                
                if (moment_inertia_handle.data[body].z == 0.0)
                    dof_one--;
                }
            else 
                {
                dof_one = 3;
                if (moment_inertia_handle.data[body].z == 0.0)
                    dof_one--;
                }
            
            query_group_dof += dof_one;
            }
        }
    
    return query_group_dof;  
    }
    
/* Compute the body forces and torques once all the particle forces are computed
    \param timestep Current time step

*/
void TwoStepNVERigid::computeForceAndTorque(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Rigid force and torque summing");
    
    // access net force data
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GPUArray< Scalar4 >& net_torque = m_pdata->getNetTorqueArray();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_torque(net_torque, access_location::host, access_mode::read);
    
    // rigid data handles
    ArrayHandle<unsigned int> body_size_handle(m_rigid_data->getBodySize(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> particle_indices_handle(m_rigid_data->getParticleIndices(), access_location::host, access_mode::read);
    unsigned int indices_pitch = m_rigid_data->getParticleIndices().getPitch();
    ArrayHandle<Scalar4> particle_pos_handle(m_rigid_data->getParticlePos(), access_location::host, access_mode::read);
    unsigned int particle_pos_pitch = m_rigid_data->getParticlePos().getPitch();
    
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::readwrite);
    
    // reset all forces and torques
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
            
        force_handle.data[body].x = 0.0;
        force_handle.data[body].y = 0.0;
        force_handle.data[body].z = 0.0;
        
        torque_handle.data[body].x = 0.0;
        torque_handle.data[body].y = 0.0;
        torque_handle.data[body].z = 0.0;
        }
            
    // for each body
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
        // for each particle
        unsigned int len = body_size_handle.data[body];
        for (unsigned int j = 0; j < len; j++)
            {
            // get the actual index of particle in the particle arrays
            unsigned int pidx = particle_indices_handle.data[body * indices_pitch + j];
            
            // access the force on the particle
            Scalar fx = h_net_force.data[pidx].x;
            Scalar fy = h_net_force.data[pidx].y;
            Scalar fz = h_net_force.data[pidx].z;

            /*Access Torque elements from a single particle. Right now I will am assuming that the particle 
              and rigid body reference frames are the same. Probably have to rotate first.
            */
            Scalar tx = h_net_torque.data[pidx].x;
            Scalar ty = h_net_torque.data[pidx].y;
            Scalar tz = h_net_torque.data[pidx].z;

            force_handle.data[body].x += fx;
            force_handle.data[body].y += fy;
            force_handle.data[body].z += fz;

            // torque = r x f
            unsigned int localidx = body * particle_pos_pitch + j;
            Scalar rx = ex_space_handle.data[body].x * particle_pos_handle.data[localidx].x
                    + ey_space_handle.data[body].x * particle_pos_handle.data[localidx].y
                    + ez_space_handle.data[body].x * particle_pos_handle.data[localidx].z;
            Scalar ry = ex_space_handle.data[body].y * particle_pos_handle.data[localidx].x
                    + ey_space_handle.data[body].y * particle_pos_handle.data[localidx].y
                    + ez_space_handle.data[body].y * particle_pos_handle.data[localidx].z;
            Scalar rz = ex_space_handle.data[body].z * particle_pos_handle.data[localidx].x
                    + ey_space_handle.data[body].z * particle_pos_handle.data[localidx].y
                    + ez_space_handle.data[body].z * particle_pos_handle.data[localidx].z;
            
            torque_handle.data[body].x += ry * fz - rz * fy + tx;
            torque_handle.data[body].y += rz * fx - rx * fz + ty;
            torque_handle.data[body].z += rx * fy - ry * fx + tz;
            }
        }
        
    if (m_prof)
        m_prof->pop();
    }

/*! Checks that every particle in the group is valid. This method may be called by anyone wishing to make this
    error check.

    TwoStepNVERigid acts as the base class for all rigid body integration methods. Check here that all particles belong
    to rigid bodies.
*/
void TwoStepNVERigid::validateGroup()
    {
    for (unsigned int gidx = 0; gidx < m_group->getNumMembers(); gidx++)
        {
        unsigned int tag = m_group->getMemberTag(gidx);
        if (m_pdata->getBody(tag) == NO_BODY)
            {
            m_exec_conf->msg->error() << "integreate.*_rigid: Particle " << tag << " does not belong to a rigid body. "
                 << "This integration method does not operate on free particles." << endl;

            throw std::runtime_error("Error initializing integration method");
            }
        }
    }

/*! Update Nose-Hoover thermostats
    \param akin_t Translational kinetic energy
    \param akin_r Rotational kinetic energy
    \param timestep Current time step
*/
void TwoStepNVERigid::update_nhcp(Scalar akin_t, Scalar akin_r, unsigned int timestep)
    {
    Scalar kt, gfkt_t, gfkt_r, tmp, ms, s, s2;
    Scalar dtv;
    
    dtv = m_deltaT;
    kt = boltz * m_temperature->getValue(timestep);
    gfkt_t = nf_t * kt;
    gfkt_r = nf_r * kt;
    
    // update thermostat masses
    
    Scalar t_mass = boltz * m_temperature->getValue(timestep) / (t_freq * t_freq);
    q_t[0] = nf_t * t_mass;
    q_r[0] = nf_r * t_mass;
    for (unsigned int i = 1; i < chain; i++)
        q_t[i] = q_r[i] = t_mass;
                
    // update force of thermostats coupled to particles
    
    f_eta_t[0] = (akin_t - gfkt_t) / q_t[0];
    f_eta_r[0] = (akin_r - gfkt_r) / q_r[0];
    
    // multiple timestep iteration
    
    for (unsigned int i = 0; i < iter; i++)
        {
        for (unsigned int j = 0; j < order; j++)
            {
            
            // update thermostat velocities half step
            
            eta_dot_t[chain-1] += wdti2[j] * f_eta_t[chain-1];
            eta_dot_r[chain-1] += wdti2[j] * f_eta_r[chain-1];
            
            for (unsigned int k = 1; k < chain; k++)
                {
                tmp = wdti4[j] * eta_dot_t[chain-k];
                ms = maclaurin_series(tmp);
                s = exp(-1.0 * tmp);
                s2 = s * s;
                eta_dot_t[chain-k-1] = eta_dot_t[chain-k-1] * s2 + 
                                       wdti2[j] * f_eta_t[chain-k-1] * s * ms;
                
                tmp = wdti4[j] * eta_dot_r[chain-k];
                ms = maclaurin_series(tmp);
                s = exp(-1.0 * tmp);
                s2 = s * s;
                eta_dot_r[chain-k-1] = eta_dot_r[chain-k-1] * s2 + 
                                       wdti2[j] * f_eta_r[chain-k-1] * s * ms;
                }
                
            // update thermostat positions a full step
            
            for (unsigned int k = 0; k < chain; k++)
                {
                eta_t[k] += wdti1[j] * eta_dot_t[k];
                eta_r[k] += wdti1[j] * eta_dot_r[k];
                }
                
            // update thermostat forces
            
            for (unsigned int k = 1; k < chain; k++)
                {
                f_eta_t[k] = q_t[k-1] * eta_dot_t[k-1] * eta_dot_t[k-1] - kt;
                f_eta_t[k] /= q_t[k];
                f_eta_r[k] = q_r[k-1] * eta_dot_r[k-1] * eta_dot_r[k-1] - kt;
                f_eta_r[k] /= q_r[k];
                }
                
            // update thermostat velocities a full step
            
            for (unsigned int k = 0; k < chain-1; k++)
                {
                tmp = wdti4[j] * eta_dot_t[k+1];
                ms = maclaurin_series(tmp);
                s = exp(-1.0 * tmp);
                s2 = s * s;
                eta_dot_t[k] = eta_dot_t[k] * s2 + wdti2[j] * f_eta_t[k] * s * ms;
                tmp = q_t[k] * eta_dot_t[k] * eta_dot_t[k] - kt;
                f_eta_t[k+1] = tmp / q_t[k+1];
                
                tmp = wdti4[j] * eta_dot_r[k+1];
                ms = maclaurin_series(tmp);
                s = exp(-1.0 * tmp);
                s2 = s * s;
                eta_dot_r[k] = eta_dot_r[k] * s2 + wdti2[j] * f_eta_r[k] * s * ms;
                tmp = q_r[k] * eta_dot_r[k] * eta_dot_r[k] - kt;
                f_eta_r[k+1] = tmp / q_r[k+1];
                }
                
            eta_dot_t[chain-1] += wdti2[j] * f_eta_t[chain-1];
            eta_dot_r[chain-1] += wdti2[j] * f_eta_r[chain-1];
            }
        }
        
    }

/*! Update Nose-Hoover thermostats coupled with barostat
    \param timestep Current time step
*/
void TwoStepNVERigid::update_nhcb(unsigned int timestep)
    {
    Scalar kt, tmp, ms, s, s2;
    Scalar dt_half;
    
    dt_half = 0.5 * m_deltaT;
    
    if (t_stat) kt = boltz * m_temperature->getValue(timestep);
    else kt = boltz;
    
    // update thermostat masses
    
    double tb_mass = kt / (p_freq * p_freq);
    q_b[0] = dimension * dimension * tb_mass;
    for (unsigned int i = 1; i < chain; i++) 
        q_b[i] = tb_mass;

    // update forces acting on thermostat
    W = (g_f + dimension) * kt / (p_freq * p_freq);
    tmp = W * epsilon_dot * epsilon_dot;
    f_eta_b[0] = (tmp - kt) / q_b[0];

    // update thermostat velocities a half step

    eta_dot_b[chain-1] += 0.5f * dt_half * f_eta_b[chain-1];

    for (unsigned int k = 0; k < chain-1; k++) 
        {
        tmp = dt_half * eta_dot_b[chain-k-1];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_b[chain-k-2] = eta_dot_b[chain-k-2] * s2 + 
                               dt_half * f_eta_b[chain-k-2] * s * ms;
        }

    // update thermostat positions

    for (unsigned int k = 0; k < chain; k++)
        eta_b[k] += m_deltaT * eta_dot_b[k];

    // update epsilon dot
  
    s = exp(-1.0 * dt_half * eta_dot_b[0]);
    epsilon_dot *= s;

    // update thermostat forces

    tmp = W * epsilon_dot * epsilon_dot;
    f_eta_b[0] = (tmp - kt) / q_b[0];
    for (unsigned int k = 1; k < chain; k++) 
        {
        f_eta_b[k] = q_b[k-1] * eta_dot_b[k-1] * eta_dot_b[k-1] - kt;
        f_eta_b[k] /= q_b[k];
        }

    // update thermostat velocites a full step

    for (unsigned int k = 0; k < chain-1; k++) 
        {
        tmp = dt_half * eta_dot_b[k+1];
        ms = maclaurin_series(tmp);
        s = exp(-0.5 * tmp);
        s2 = s * s;
        eta_dot_b[k] = eta_dot_b[k] * s2 + dt_half * f_eta_b[k] * s * ms;
        tmp = q_b[k] * eta_dot_b[k] * eta_dot_b[k] - kt;
        f_eta_b[k+1] = tmp / q_b[k+1];
        }

    eta_dot_b[chain-1] += 0.5f * dt_half * f_eta_b[chain-1];
    }

/*! Calculate the new box size from dilation
    Remap the rigid body COMs from old box to new box
    Note that NPT rigid currently only deals with rigid bodies, no point particles
    For hybrid systems, use TwoStepNPT coupled with TwoStepNVTRigid to avoid duplicating box resize
*/
void TwoStepNVERigid::remap()
{
    Scalar oldlo, oldhi, ctr;
    Scalar invLx, invLy, invLz;
    
    BoxDim box = m_pdata->getBox();
    
    // convert rigid body COMs to lamda coords

    ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::readwrite);
    
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();
    Scalar3 L = box.getL();
    invLx = 1.0 / L.x;
    invLy = 1.0 / L.y;
    invLz = 1.0 / L.z;
    
    Scalar4 delta;
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
            
        delta.x = com_handle.data[body].x - lo.x;
        delta.y = com_handle.data[body].y - lo.y;
        delta.z = com_handle.data[body].z - lo.z;

        com_handle.data[body].x = invLx * delta.x;
        com_handle.data[body].y = invLy * delta.y;
        com_handle.data[body].z = invLz * delta.z;
        }

    // reset box to new size/shape
    oldlo = lo.x;
    oldhi = hi.x;
    ctr = 0.5 * (oldlo + oldhi);
    lo.x = (oldlo - ctr) * dilation + ctr;
    hi.x = (oldhi - ctr) * dilation + ctr;
    L.x = hi.x - lo.x;
    
    oldlo = lo.y;
    oldhi = hi.y;
    ctr = 0.5 * (oldlo + oldhi);
    lo.y = (oldlo - ctr) * dilation + ctr;
    hi.y = (oldhi - ctr) * dilation + ctr;
    L.y = hi.y - lo.y;
    
    if (dimension == 3)
        {
        oldlo = lo.z;
        oldhi = hi.z;
        ctr = 0.5 * (oldlo + oldhi);
        lo.z = (oldlo - ctr) * dilation + ctr;
        hi.z = (oldhi - ctr) * dilation + ctr;
        L.z = hi.z - lo.z;
        }
    
    m_pdata->setGlobalBoxL(L);
    
    // convert rigid body COMs back to box coords
    Scalar4 newboxlo;
    newboxlo.x = -L.x/2.0;
    newboxlo.y = -L.y/2.0;
    newboxlo.z = -L.z/2.0;
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
            
        com_handle.data[body].x = L.x * com_handle.data[body].x + newboxlo.x;
        com_handle.data[body].y = L.y * com_handle.data[body].y + newboxlo.y;
        com_handle.data[body].z = L.z * com_handle.data[body].z + newboxlo.z;
        }
    
    }

void export_TwoStepNVERigid()
{
    class_<TwoStepNVERigid, boost::shared_ptr<TwoStepNVERigid>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
    ("TwoStepNVERigid", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup> >())
    .def("setT", &TwoStepNVERigid::setT)
    .def("setP", &TwoStepNVERigid::setP)
    .def("setTau", &TwoStepNVERigid::setTau)
    .def("setTauP", &TwoStepNVERigid::setTauP)
    .def("setPartialScale", &TwoStepNVERigid::setPartialScale)
    ;
}

#ifdef WIN32
#pragma warning( pop )
#endif

