/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

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

#include "QuaternionMath.h"
#include "TwoStepNVERigid.h"
#include <boost/python.hpp>

using namespace std;
using namespace boost::python;

/*! \file TwoStepNVERigid.cc
 \brief Defines the TwoStepNVERigid class: standard rigid body dynamics without conjugate quaternions
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param suffix Suffix to attach to the end of log quantity names
 */
TwoStepNVERigid::TwoStepNVERigid(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<ParticleGroup> group)
    : TwoStepNHRigid(sysdef, group, std::string(""), 0, 0, 0)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVERigid" << endl;

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

TwoStepNVERigid::~TwoStepNVERigid()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVERigid" << endl;
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

    if (m_prof)
        m_prof->push("NVE rigid step 2");

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

void export_TwoStepNVERigid()
{
    class_<TwoStepNVERigid, boost::shared_ptr<TwoStepNVERigid>, bases<TwoStepNHRigid>, boost::noncopyable>
    ("TwoStepNVERigid", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup> >())
    ;
}

