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

#include "FIREEnergyMinimizerRigid.h"
#include "TwoStepNVERigid.h"

/*! \file FIREEnergyMinimizerRigid.h
    \brief Contains code for the FIREEnergyMinimizerRigid class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on (group is a placeholder for now)
    \param dt Time step for MD integrator
    \param reset_and_create_integrator Flag to indicate if resetting and creating integrator are needed 
    \post The method is constructed with the given particle data and a NULL profiler.
*/
FIREEnergyMinimizerRigid::FIREEnergyMinimizerRigid(boost::shared_ptr<SystemDefinition> sysdef,
                                                    boost::shared_ptr<ParticleGroup> group,
                                                    Scalar dt, 
                                                    bool reset_and_create_integrator)
    :   FIREEnergyMinimizer(sysdef, group, dt, false), m_wtol(Scalar(1e-1)) // using false for the parent class
    {

    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    m_nparticles = arrays.nparticles;
    m_pdata->release();
    
    // Get the system rigid data
    m_rigid_data = sysdef->getRigidData();
    
    // Create my rigid body group from the particle group
    m_body_group = boost::shared_ptr<RigidBodyGroup>(new RigidBodyGroup(sysdef, m_group));
    
    // Get the number of rigid bodies for frequent use
    m_n_bodies = m_body_group->getNumMembers();
    
    if (m_n_bodies == 0)
        {
        cout << "***Warning! Empty group of rigid bodies." << endl;
        }
    
    // Time steps to run NVE between minimizer moves    
    m_nevery = 1;
        
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    if (reset_and_create_integrator)
        {
        reset();
        boost::shared_ptr<TwoStepNVERigid> integrator(new TwoStepNVERigid(sysdef, group));
        addIntegrationMethod(integrator);
        setDeltaT(m_deltaT);
        }
    }

/*! Reset minimizer parameters and zero velocities
*/
void FIREEnergyMinimizerRigid::reset()
    {
    m_converged = false;
    m_n_since_negative = m_nmin+1;
     m_n_since_start = 0;    
    m_alpha = m_alpha_start;
    m_was_reset = true;
    
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
            
        // scales translational velocity
        vel_handle.data[body].x = 0.0;
        vel_handle.data[body].y = 0.0;
        vel_handle.data[body].z = 0.0;
        
        angmom_handle.data[body].x = 0.0;
        angmom_handle.data[body].y = 0.0;
        angmom_handle.data[body].z = 0.0;
        }
        
    setDeltaT(m_deltaT_set);
    }

        
/*! \param timestep is the current timestep
*/
void FIREEnergyMinimizerRigid::update(unsigned int timestep)
    {
    if (m_converged)
        return;
    
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
        
    IntegratorTwoStep::update(timestep);
    
    if (timestep % m_nevery != 0)
        return;
        
    if (m_n_bodies <= 0)
        {
        cerr << endl << "***Error! FIREENergyMinimizerRigid: There is no rigid body for this integrator" << endl << endl;
        throw runtime_error("Error update for FIREEnergyMinimizerRigid (no rigid body)");
        return;
        }
        
    Scalar Pt(0.0), Pr(0.0);
    Scalar vnorm(0.0), wnorm(0.0);
    Scalar fnorm(0.0), tnorm(0.0);
    
    // Calculate the per-particle potential energy over particles in the group
    Scalar energy = 0.0;
    
    {
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // total potential energy 
    double pe_total = 0.0;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        pe_total += (double)h_net_force.data[j].w;
        }
    energy = pe_total/Scalar(group_size);    
    }
    
    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000) * m_etol;
        }
   
    
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::read);
    
    // Calculates the powers
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
            
        // translational power = force * vel
        Pt += force_handle.data[body].x * vel_handle.data[body].x + force_handle.data[body].y * vel_handle.data[body].y 
                + force_handle.data[body].z * vel_handle.data[body].z;
        fnorm += force_handle.data[body].x * force_handle.data[body].x + force_handle.data[body].y * force_handle.data[body].y 
                + force_handle.data[body].z * force_handle.data[body].z;
        vnorm += vel_handle.data[body].x * vel_handle.data[body].x + vel_handle.data[body].y * vel_handle.data[body].y 
                + vel_handle.data[body].z * vel_handle.data[body].z;
        
        // rotational power = torque * angvel
        Pr += torque_handle.data[body].x * angvel_handle.data[body].x + torque_handle.data[body].y * angvel_handle.data[body].y 
                + torque_handle.data[body].z * angvel_handle.data[body].z;
        tnorm += torque_handle.data[body].x * torque_handle.data[body].x + torque_handle.data[body].y * torque_handle.data[body].y 
                + torque_handle.data[body].z * torque_handle.data[body].z;
        wnorm += angvel_handle.data[body].x * angvel_handle.data[body].x + angvel_handle.data[body].y * angvel_handle.data[body].y 
                + angvel_handle.data[body].z * angvel_handle.data[body].z;
        }
    
    fnorm = sqrt(fnorm);
    vnorm = sqrt(vnorm);
    
    tnorm = sqrt(tnorm);
    wnorm = sqrt(wnorm);
    
    //printf("f = %g (%g); w = %g (%g); e = %g (%g); min_steps: %d (%d) \n", fnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies), m_ftol, wnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies), m_wtol, fabs(energy-m_old_energy), m_etol, m_n_since_start, m_run_minsteps);

    if ((fnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies) < m_ftol && wnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies) < m_wtol  && fabs(energy-m_old_energy) < m_etol) && m_n_since_start >= m_run_minsteps)
        {
        printf("Converged: f = %g (ftol = %g); w= %g (wtol = %g); e = %g (etol = %g)\n", fnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies), m_ftol, wnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies), m_wtol, fabs(energy-m_old_energy), m_etol);
        m_converged = true;
        return;
        }

    // Scales velocities and angular momenta
    Scalar factor_t, factor_r;
    if (fabs(fnorm) > EPSILON)
        factor_t = m_alpha * vnorm / fnorm;
    else
        factor_t = 1.0; 
        
    if (fabs(tnorm) > EPSILON)    
        factor_r = m_alpha * wnorm / tnorm;
    else 
        factor_r = 1.0; 
        
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
            
        // scales translational velocity
        vel_handle.data[body].x = vel_handle.data[body].x * (1.0 - m_alpha) + force_handle.data[body].x * factor_t;
        vel_handle.data[body].y = vel_handle.data[body].y * (1.0 - m_alpha) + force_handle.data[body].y * factor_t;
        vel_handle.data[body].z = vel_handle.data[body].z * (1.0 - m_alpha) + force_handle.data[body].z * factor_t;
        
        // scales angular momenta (now using the same m_alpha as translational)
        angmom_handle.data[body].x = angmom_handle.data[body].x * (1.0 - m_alpha) + torque_handle.data[body].x * factor_r;
        angmom_handle.data[body].y = angmom_handle.data[body].y * (1.0 - m_alpha) + torque_handle.data[body].y * factor_r;
        angmom_handle.data[body].z = angmom_handle.data[body].z * (1.0 - m_alpha) + torque_handle.data[body].z * factor_r;
        }
    
    // A simply naive measure is to sum up the power coming from translational and rotational motions,
    // more sophisticated measure can be devised later
    Scalar P = Pt + Pr;
    if (P > Scalar(0.0))
        {
        m_n_since_negative++;
        if (m_n_since_negative > m_nmin) // has been positive for a while?
            {
            IntegratorTwoStep::setDeltaT(std::min(m_deltaT * m_finc, m_deltaT_max));
            m_alpha *= m_falpha; // decreases m_alpha to reduce the influence of the power on the velocities in searching for the well
            }
        }
    else if (P <= Scalar(0.0))
        {
        IntegratorTwoStep::setDeltaT(m_deltaT * m_fdec);
        m_alpha = m_alpha_start;
        m_n_since_negative = 0;
        for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
            {
            unsigned int body = m_body_group->getMemberIndex(group_idx);
            
            vel_handle.data[body].x = Scalar(0.0);
            vel_handle.data[body].y = Scalar(0.0);
            vel_handle.data[body].z = Scalar(0.0);
            
            angmom_handle.data[body].x = Scalar(0.0);
            angmom_handle.data[body].y = Scalar(0.0);
            angmom_handle.data[body].z = Scalar(0.0);
            }
        }

    m_n_since_start++;    
    m_old_energy = energy;
    }


void export_FIREEnergyMinimizerRigid()
    {
    class_<FIREEnergyMinimizerRigid, boost::shared_ptr<FIREEnergyMinimizerRigid>, bases<FIREEnergyMinimizer>, boost::noncopyable>
        ("FIREEnergyMinimizerRigid", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar >())
        .def("setWtol", &FIREEnergyMinimizerRigid::setWtol)        
        .def("getEvery", &FIREEnergyMinimizerRigid::getEvery)
        .def("setEvery", &FIREEnergyMinimizerRigid::setEvery);
    }

#ifdef WIN32
#pragma warning( pop )
#endif

