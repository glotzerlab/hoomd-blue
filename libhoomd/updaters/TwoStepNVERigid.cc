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
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "TwoStepNVERigid.h"
#include <math.h>
#include <fstream>

//! Absolute threshold for zero moment of inertia
#define EPSILON 1e-3

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
    m_n_bodies = m_rigid_data->getNumBodies();
    
    // sanity check
    if (m_n_bodies <= 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    
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
        ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::readwrite);
                
        ArrayHandle<bool> angmom_init_handle(m_rigid_data->getAngMomInit(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
        
        // Reset all forces and torques
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
            bool angmom_init = angmom_init_handle.data[body];
            
            vel_handle.data[body].x = 0.0;
            vel_handle.data[body].y = 0.0;
            vel_handle.data[body].z = 0.0;
            
            force_handle.data[body].x = 0.0;
            force_handle.data[body].y = 0.0;
            force_handle.data[body].z = 0.0;
            
            torque_handle.data[body].x = 0.0;
            torque_handle.data[body].y = 0.0;
            torque_handle.data[body].z = 0.0;
            
            if (angmom_init == false) // needs to initialize
                {
                angmom_handle.data[body].x = 0.0;
                angmom_handle.data[body].y = 0.0;
                angmom_handle.data[body].z = 0.0;
                }
            }
            
        // Access the particle data arrays
        ParticleDataArrays arrays = m_pdata->acquireReadWrite();
        
        // for each body
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
            bool angmom_init = angmom_init_handle.data[body];
            
            // for each particle
            unsigned int len = body_size_handle.data[body];
            for (unsigned int j = 0; j < len; j++)
                {
                // get the index of particle in the particle arrays
                unsigned int pidx = particle_indices_handle.data[body * indices_pitch + j];
                
                // get the particle mass
                Scalar mass_one = arrays.mass[pidx];
                
                vel_handle.data[body].x += mass_one * arrays.vx[pidx];
                vel_handle.data[body].y += mass_one * arrays.vy[pidx];
                vel_handle.data[body].z += mass_one * arrays.vz[pidx];
                
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
                
                torque_handle.data[body].x += ry * fz - rz * fy;
                torque_handle.data[body].y += rz * fx - rx * fz;
                torque_handle.data[body].z += rx * fy - ry * fx;
              
                // Angular momentum = r x (m * v) is calculated for setup
                if (angmom_init == false) // if angmom is not yet set for this body
                    {
                    angmom_handle.data[body].x += ry * (mass_one * arrays.vz[pidx]) - rz * (mass_one * arrays.vy[pidx]);
                    angmom_handle.data[body].y += rz * (mass_one * arrays.vx[pidx]) - rx * (mass_one * arrays.vz[pidx]);
                    angmom_handle.data[body].z += rx * (mass_one * arrays.vy[pidx]) - ry * (mass_one * arrays.vx[pidx]);
                    }
                }
            
            }
            
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
            vel_handle.data[body].x /= body_mass_handle.data[body];
            vel_handle.data[body].y /= body_mass_handle.data[body];
            vel_handle.data[body].z /= body_mass_handle.data[body];
            
            computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body],
                                   ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angvel_handle.data[body]);
            }
            
        m_pdata->release();

        } // out of scope for handles   
    
    // Set the velocities of particles in rigid bodies
    set_v(0);
/*
#define __DEBUG
#ifdef __DEBUG
    {
    ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::readwrite);
                
    std::ofstream ofs("bodies.txt");
    for (unsigned int body = 0; body < m_n_bodies; body++)
    {
        ofs << "body " << body << "\n";
        ofs << "angvel = " << angvel_handle.data[body].x << "\t" << angvel_handle.data[body].y << "\t" << angvel_handle.data[body].z << "\n";
        ofs << "force = " << force_handle.data[body].x << "\t" << force_handle.data[body].y << "\t" << force_handle.data[body].z << "\n";
        ofs << "torque = " << torque_handle.data[body].x << "\t" << torque_handle.data[body].y << "\t" << torque_handle.data[body].z << "\n";
        ofs << "angmom = " << angmom_handle.data[body].x << "\t" << angmom_handle.data[body].y << "\t" << angmom_handle.data[body].z << "\n";
        
        
    }
    ofs.close();
    }
#endif
#undef __DEBUG
*/
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
    Scalar dt_half = 0.5 * m_deltaT;
    Scalar dtfm;
    
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
        
        // for each body
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
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
                
            angmom_handle.data[body].x += dt_half * torque_handle.data[body].x;
            angmom_handle.data[body].y += dt_half * torque_handle.data[body].y;
            angmom_handle.data[body].z += dt_half * torque_handle.data[body].z;
            
            advanceQuaternion(angmom_handle.data[body], moment_inertia_handle.data[body], angvel_handle.data[body],
                              ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], orientation_handle.data[body]);
            }
        } // out of scope for handles
        
    // set positions and velocities of particles in rigid bodies
    set_xv(timestep);

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
/*    
#define __DEBUG
#ifdef __DEBUG
    {
    ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::readwrite);
                
    std::ofstream ofs("bodies1.txt");
    for (unsigned int body = 0; body < m_n_bodies; body++)
    {
        ofs << "body " << body << "\n";
        ofs << "angvel = " << angvel_handle.data[body].x << "\t" << angvel_handle.data[body].y << "\t" << angvel_handle.data[body].z << "\n";
        ofs << "force = " << force_handle.data[body].x << "\t" << force_handle.data[body].y << "\t" << force_handle.data[body].z << "\n";
        ofs << "torque = " << torque_handle.data[body].x << "\t" << torque_handle.data[body].y << "\t" << torque_handle.data[body].z << "\n";
        ofs << "angmom = " << angmom_handle.data[body].x << "\t" << angmom_handle.data[body].y << "\t" << angmom_handle.data[body].z << "\n";
        
        
    }
    ofs.close();
    }
#endif
//#undef __DEBUG
*/
    if (m_prof)
        m_prof->push("NVE rigid step 2");
    
        {
        // rigid data handes
        ArrayHandle<Scalar> body_mass_handle(m_rigid_data->getBodyMass(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> moment_inertia_handle(m_rigid_data->getMomentInertia(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ex_space_handle(m_rigid_data->getExSpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ey_space_handle(m_rigid_data->getEySpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ez_space_handle(m_rigid_data->getEzSpace(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::readwrite);
        
        
        Scalar dt_half = 0.5 * m_deltaT;
        
        // 2nd step: final integration
        for (unsigned int body = 0; body < m_n_bodies; body++)
            {
            Scalar dtfm = dt_half / body_mass_handle.data[body];
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

    // set velocities of particles in rigid bodies
    set_v(timestep);

    if (m_prof)
        m_prof->pop();
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
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
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
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
        force_handle.data[body].x = 0.0;
        force_handle.data[body].y = 0.0;
        force_handle.data[body].z = 0.0;
        
        torque_handle.data[body].x = 0.0;
        torque_handle.data[body].y = 0.0;
        torque_handle.data[body].z = 0.0;
        }
            
    // for each body
    for (unsigned int body = 0; body < m_n_bodies; body++)
        {
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
            
            torque_handle.data[body].x += ry * fz - rz * fy;
            torque_handle.data[body].y += rz * fx - rx * fz;
            torque_handle.data[body].z += rx * fy - ry * fx;
            }
        }
    
    //m_pdata->release();
    
    if (m_prof)
        m_prof->pop();
    }

/* Set position and velocity of constituent particles in rigid bodies in the 1st half of integration
    based on the body center of mass and particle relative position in each body frame.
    \param timestep Current time step
*/

void TwoStepNVERigid::set_xv(unsigned int timestep)
    {
    // get box
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
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
                
            // v_particle = v_com + angvel x xr
            arrays.vx[pidx] = vel_handle.data[body].x + angvel_handle.data[body].y * zr - angvel_handle.data[body].z * yr;
            arrays.vy[pidx] = vel_handle.data[body].y + angvel_handle.data[body].z * xr - angvel_handle.data[body].x * zr;
            arrays.vz[pidx] = vel_handle.data[body].z + angvel_handle.data[body].x * yr - angvel_handle.data[body].y * xr;
            
            }
        }
        
    m_pdata->release();
    
    }

/* Set velocity of constituent particles in rigid bodies in the 2nd half of integration
 based on the body center of mass and particle relative position in each body frame.
    \param timestep Current time step
*/

void TwoStepNVERigid::set_v(unsigned int timestep)
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
    
    // access the particle data arrays
    ParticleDataArrays arrays = m_pdata->acquireReadWrite();
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
                        
            // v_particle = v_com + angvel x xr
            arrays.vx[pidx] = vel_handle.data[body].x + angvel_handle.data[body].y * zr - angvel_handle.data[body].z * yr;
            arrays.vy[pidx] = vel_handle.data[body].y + angvel_handle.data[body].z * xr - angvel_handle.data[body].x * zr;
            arrays.vz[pidx] = vel_handle.data[body].z + angvel_handle.data[body].x * yr - angvel_handle.data[body].y * xr;
            }
        }
        
    m_pdata->release();
    
    }

/* Compute orientation (ex_space, ey_space, ez_space) from quaternion- re-implement from RigidData for self-containing purposes
    \param quat Quaternion
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
*/

void TwoStepNVERigid::exyzFromQuaternion(Scalar4 &quat, Scalar4 &ex_space, Scalar4 &ey_space, Scalar4 &ez_space)
    {
    // ex_space
    ex_space.x = quat.x * quat.x + quat.y * quat.y - quat.z * quat.z - quat.w * quat.w;
    ex_space.y = 2.0 * (quat.y * quat.z + quat.x * quat.w);
    ex_space.z = 2.0 * (quat.y * quat.w - quat.x * quat.z);
    
    // ey_space
    ey_space.x = 2.0 * (quat.y * quat.z - quat.x * quat.w);
    ey_space.y = quat.x * quat.x - quat.y * quat.y + quat.z * quat.z - quat.w * quat.w;
    ey_space.z = 2.0 * (quat.z * quat.w + quat.x * quat.y);
    
    // ez_space
    ez_space.x = 2.0 * (quat.y * quat.w + quat.x * quat.z);
    ez_space.y = 2.0 * (quat.z * quat.w - quat.x * quat.y);
    ez_space.z = quat.x * quat.x - quat.y * quat.y - quat.z * quat.z + quat.w * quat.w;
    }

// Compute angular velocity from angular momentum
/*!  Convert the angular momentum from world frame to body frame.
    Compute angular velocity in the body frame (angbody).
    Convert the angular velocity from body frame back to world frame.

    Rotation matrix is formed by arranging ex_space, ey_space and ez_space vectors into columns.
    In this code, rotation matrix is used to map a vector in a body frame into the space frame:
        x_space = rotation_matrix * x_body
    The reverse operation is to convert a vector in the space frame to a body frame:
        x_body = transpose(rotation matrix) * x_space
    
    \param angmom Angular momentum
    \param moment_inertia Moment of inertia
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param angvel Returned angular velocity
 
 */

void TwoStepNVERigid::computeAngularVelocity(Scalar4& angmom, Scalar4& moment_inertia, Scalar4& ex_space, Scalar4& ey_space, Scalar4& ez_space,
                                             Scalar4& angvel)
    {
    //! Angular velocity in the body frame
    Scalar angbody[3];
    
    //! angbody = angmom_body / moment_inertia = transpose(rotation_matrix) * angmom / moment_inertia
    if (moment_inertia.x < EPSILON) angbody[0] = 0.0;
    else angbody[0] = (ex_space.x * angmom.x + ex_space.y * angmom.y
                           + ex_space.z * angmom.z) / moment_inertia.x;
                           
    if (moment_inertia.y < EPSILON) angbody[1] = 0.0;
    else angbody[1] = (ey_space.x * angmom.x + ey_space.y * angmom.y
                           + ey_space.z * angmom.z) / moment_inertia.y;
                           
    if (moment_inertia.z < EPSILON) angbody[2] = 0.0;
    else angbody[2] = (ez_space.x * angmom.x + ez_space.y * angmom.y
                           + ez_space.z * angmom.z) / moment_inertia.z;
                           
    //! Convert to angbody to the space frame: angvel = rotation_matrix * angbody
    angvel.x = angbody[0] * ex_space.x + angbody[1] * ey_space.x + angbody[2] * ez_space.x;
    angvel.y = angbody[0] * ex_space.y + angbody[1] * ey_space.y + angbody[2] * ez_space.y;
    angvel.z = angbody[0] * ex_space.z + angbody[1] * ey_space.z + angbody[2] * ez_space.z;
    }

// Advance the quaternion using angular momentum and angular velocity
/*  \param angmom Angular momentum
    \param moment_inerta Moment of inertia
    \param angvel Returned angular velocity
    \param ex_space x-axis unit vector
    \param ey_space y-axis unit vector
    \param ez_space z-axis unit vector
    \param quat Returned quaternion

*/
void TwoStepNVERigid::advanceQuaternion(Scalar4& angmom, Scalar4 &moment_inertia, Scalar4 &angvel,
                                        Scalar4& ex_space, Scalar4& ey_space, Scalar4& ez_space, Scalar4 &quat)
    {
    Scalar4 qhalf, qfull, omegaq;
    Scalar dtq = 0.5 * m_deltaT;
    
    computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);
    
    // Compute (w q)
    multiply(angvel, quat, omegaq);
    
    // Full update q from dq/dt = 1/2 w q
    qfull.x = quat.x + dtq * omegaq.x;
    qfull.y = quat.y + dtq * omegaq.y;
    qfull.z = quat.z + dtq * omegaq.z;
    qfull.w = quat.w + dtq * omegaq.w;
    normalize(qfull);
    
    // 1st half update from dq/dt = 1/2 w q
    qhalf.x = quat.x + 0.5 * dtq * omegaq.x;
    qhalf.y = quat.y + 0.5 * dtq * omegaq.y;
    qhalf.z = quat.z + 0.5 * dtq * omegaq.z;
    qhalf.w = quat.w + 0.5 * dtq * omegaq.w;
    normalize(qhalf);
    
    // Udpate ex, ey, ez from qhalf = update A
    exyzFromQuaternion(qhalf, ex_space, ey_space, ez_space);
    
    // Compute angular velocity from new ex_space, ey_space and ex_space
    computeAngularVelocity(angmom, moment_inertia, ex_space, ey_space, ez_space, angvel);
    
    // Compute (w qhalf)
    multiply(angvel, qhalf, omegaq);
    
    // 2nd half update from dq/dt = 1/2 w q
    qhalf.x += 0.5 * dtq * omegaq.x;
    qhalf.y += 0.5 * dtq * omegaq.y;
    qhalf.z += 0.5 * dtq * omegaq.z;
    qhalf.w += 0.5 * dtq * omegaq.w;
    normalize(qhalf);
    
    // Corrected Richardson update
    quat.x = 2.0 * qhalf.x - qfull.x;
    quat.y = 2.0 * qhalf.y - qfull.y;
    quat.z = 2.0 * qhalf.z - qfull.z;
    quat.w = 2.0 * qhalf.w - qfull.w;
    normalize(quat);
    
    exyzFromQuaternion(quat, ex_space, ey_space, ez_space);
    }

// Quaternion multiply: c = a * b where a = (0, a)
/*  \param a Quaternion
    \param b Quaternion
    \param c Returned quaternion
*/

void TwoStepNVERigid::multiply(Scalar4 &a, Scalar4 &b, Scalar4 &c)
    {
    c.x = -(a.x * b.y + a.y * b.z + a.z * b.w);
    c.y =   b.x * a.x + a.y * b.w - a.z * b.z;
    c.z =   b.x * a.y + a.z * b.y - a.x * b.w;
    c.w =   b.x * a.z + a.x * b.z - a.y * b.y;
    }

/*! 
    \param q Quaternion to be normalized
 */

void TwoStepNVERigid::normalize(Scalar4 &q)
    {
    Scalar norm = 1.0 / sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q.x *= norm;
    q.y *= norm;
    q.z *= norm;
    q.w *= norm;
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

