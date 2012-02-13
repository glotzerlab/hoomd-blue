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
        cout << "***Warning! Empty group for rigid body integration." << endl;
        }
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
     
    // sanity check
    if (m_n_bodies <= 0)
        return;
    
    //GPUArray<Scalar> virial(m_rigid_data->getNmax(), m_n_bodies, m_pdata->getExecConf());
    //m_virial.swap(virial);
    
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
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
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
            cerr << endl;
            cerr << "***Error! Particle " << tag << " does not belong to a rigid body. "
                 << "This integration method does not operate on free particles." << endl << endl;
                
            throw std::runtime_error("Error initializing integration method");
            }
        }
    }

void export_TwoStepNVERigid()
{
    class_<TwoStepNVERigid, boost::shared_ptr<TwoStepNVERigid>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
    ("TwoStepNVERigid", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup> >())
    ;
}

#ifdef WIN32
#pragma warning( pop )
#endif

