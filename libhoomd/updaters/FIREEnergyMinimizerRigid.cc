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

// $Id: FIREEnergyMinimizer.cc 2587 2010-01-08 17:02:54Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/libhoomd/updaters/FIREEnergyMinimizer.cc $
// Maintainer: joaander

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
    :   FIREEnergyMinimizer(sysdef, dt, false) // using false for the parent class
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    m_nparticles = arrays.nparticles;
    m_pdata->release();

    m_rigid_data = sysdef->getRigidData();
    
    m_nevery = 1;
    
    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    if (reset_and_create_integrator)
        {
        reset();
     //   createIntegrator();
        boost::shared_ptr<TwoStepNVERigid> integrator(new TwoStepNVERigid(sysdef, group));
        addIntegrationMethod(integrator);
        setDeltaT(m_deltaT);
        }
    }

void FIREEnergyMinimizerRigid::createIntegrator()
    {
    boost::shared_ptr<ParticleSelector> selector_rigid(new ParticleSelectorRigid(m_sysdef, true));
    boost::shared_ptr<ParticleGroup> group_rigid(new ParticleGroup(m_sysdef, selector_rigid));
    boost::shared_ptr<TwoStepNVERigid> integrator(new TwoStepNVERigid(m_sysdef, group_rigid));
    addIntegrationMethod(integrator);
    setDeltaT(m_deltaT);
    }

void FIREEnergyMinimizerRigid::reset()
    {
    m_converged = false;
    m_n_since_negative = 0;
    m_alpha = m_alpha_start;
    m_was_reset = true;
    
    unsigned int n_bodies = m_rigid_data->getNumBodies();
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::host, access_mode::readwrite);
    for (unsigned int body = 0; body < n_bodies; body++)
        {
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
    
    // The energy minimized is currently the system potential energy
    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    unsigned int nparticles = arrays.nparticles;
    Scalar energy = computePotentialEnergy(timestep) / nparticles;

    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000) * m_etol;
        }
   
    
    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::host, access_mode::read);
    
    // Calculates the powers
    for (unsigned int body = 0; body < n_bodies; body++)
        {
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
    
    
    if (fnorm/sqrt(m_sysdef->getNDimensions() * n_bodies) < m_ftol || fabs(energy-m_old_energy) < m_etol)
        {
        printf("f = %g (%g); e = %g (%g)\n", fnorm/sqrt(m_sysdef->getNDimensions() * n_bodies), m_ftol, fabs(energy-m_old_energy), m_etol);
        m_converged = true;
        return;
        }

    // Scales velocities and angular momenta
    Scalar factor_t = m_alpha * vnorm / fnorm;
    Scalar factor_r = m_alpha * wnorm / tnorm;
    for (unsigned int body = 0; body < n_bodies; body++)
        {
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
        for (unsigned int body = 0; body < n_bodies; body++)
            {
            vel_handle.data[body].x = Scalar(0.0);
            vel_handle.data[body].y = Scalar(0.0);
            vel_handle.data[body].z = Scalar(0.0);
            
            angmom_handle.data[body].x = Scalar(0.0);
            angmom_handle.data[body].y = Scalar(0.0);
            angmom_handle.data[body].z = Scalar(0.0);
            }
        }
    m_pdata->release();
    m_old_energy = energy;

    }


void export_FIREEnergyMinimizerRigid()
    {
    class_<FIREEnergyMinimizerRigid, boost::shared_ptr<FIREEnergyMinimizerRigid>, boost::noncopyable>
        ("FIREEnergyMinimizerRigid", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar >())
        .def("getEvery", &FIREEnergyMinimizerRigid::getEvery)
        .def("setEvery", &FIREEnergyMinimizerRigid::setEvery)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

