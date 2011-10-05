/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
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

// Maintainer: ndtrung

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "QuaternionMath.h"
#include "TwoStepBDNVTRigid.h"

/*! \file TwoStepBDNVTRigid.cc
    \brief Contains code for the TwoStepBDNVTRigid class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param gamma_diam Set gamma to the particle diameter of each particle if true, otherwise use a per-type
                      gamma via setGamma()
*/
TwoStepBDNVTRigid::TwoStepBDNVTRigid(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<ParticleGroup> group,
                           boost::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool gamma_diam)
    : TwoStepNVERigid(sysdef, group, true), m_T(T), m_seed(seed), m_gamma_diam(gamma_diam)
    {
    // set a named, but otherwise blank set of integrator variables
    IntegratorVariables v = getIntegratorVariables();

    if (!restartInfoTestValid(v, "bdnvt_rigid", 0))
        {
        v.type = "bdnvt_rigid";
        v.variable.resize(0);
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);

    // allocate memory for the per-type gamma storage and initialize them to 1.0
    GPUArray<Scalar> gamma(m_pdata->getNTypes(), m_pdata->getExecConf());
    m_gamma.swap(gamma);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma.getNumElements(); i++)
        h_gamma.data[i] = Scalar(1.0);
    }

/*! \param typ Particle type to set gamma for
    \param gamma The gamma value to set
*/
void TwoStepBDNVTRigid::setGamma(unsigned int typ, Scalar gamma)
    {
    // check for user errors
    if (m_gamma_diam)
        {
        cerr << endl << "***Error! Trying to set gamma when it is set to be the diameter! " << typ << endl << endl;
        throw runtime_error("Error setting params in TwoStepBDNVT");
        }
    if (typ >= m_pdata->getNTypes())
        {
        cerr << endl << "***Error! Trying to set gamma for a non existant type! " << typ << endl << endl;
        throw runtime_error("Error setting params in TwoStepBDNVT");
        }
        
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);
    h_gamma.data[typ] = gamma;
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepBDNVTRigid::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;
        
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    
    // profile this step
    if (m_prof)
        m_prof->push("BD NVT rigid step 2");

    {
    // Modify the net forces with the random and drag forces
    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);

    // grab some initial variables
    const Scalar currentTemp = m_T->getValue(timestep);
    const Scalar D = Scalar(m_sysdef->getNDimensions());
    
    // initialize the RNG
    Saru saru(m_seed, timestep);

    // a(t+deltaT) gets modified with the bd forces
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    unsigned int group_size = m_group->getNumMembers();
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        // first, calculate the BD forces
        // Generate three random numbers
        Scalar rx = saru.d(-1,1);
        Scalar ry = saru.d(-1,1);
        Scalar rz = saru.d(-1,1);
        
        Scalar gamma;
        if (m_gamma_diam)
            gamma = arrays.diameter[j];
        else
            gamma = h_gamma.data[arrays.type[j]];
        
        // compute the bd force
        Scalar coeff = sqrt(Scalar(6.0)*gamma*currentTemp/m_deltaT);
        Scalar bd_fx = rx*coeff - gamma*arrays.vx[j];
        Scalar bd_fy = ry*coeff - gamma*arrays.vy[j];
        Scalar bd_fz = rz*coeff - gamma*arrays.vz[j];
        
        if (D < 3.0)
            bd_fz = Scalar(0.0);

        h_net_force.data[j].x += bd_fx;
        h_net_force.data[j].y += bd_fy;
        h_net_force.data[j].z += bd_fz;
        }
        
    m_pdata->release();
    }
        
    // Perform the second step like in TwoStepNVERigid
    // compute net forces and torques on rigid bodies from particle forces
    computeForceAndTorque(timestep);
    
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
    
    // 2nd step: final integration
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        
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
        
    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepBDNVTRigid()
    {
    class_<TwoStepBDNVTRigid, boost::shared_ptr<TwoStepBDNVTRigid>, bases<TwoStepNVERigid>, boost::noncopyable>
        ("TwoStepBDNVTRigid", init< boost::shared_ptr<SystemDefinition>,
                         boost::shared_ptr<ParticleGroup>,
                         boost::shared_ptr<Variant>,
                         unsigned int,
                         bool
                         >())
        .def("setT", &TwoStepBDNVTRigid::setT)
        .def("setGamma", &TwoStepBDNVTRigid::setGamma)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

