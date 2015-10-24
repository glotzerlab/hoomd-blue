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
#include "TwoStepNHRigid.h"
#include <boost/python.hpp>
#include <math.h>

using namespace std;
using namespace boost::python;

/*! \file TwoStepNHRigid.cc
 \brief Defines the TwoStepNHRigid class, the base class for NVE, NVT, NPT and NPH rigid body integrators
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
 \param group The group of particles this integration method is to work on
 \param suffix Suffix to attach to the end of log quantity names
 \param tchain Number of thermostats in the thermostat chain
 \param pchain Number of thermostats coupled with the barostat
 \param iter Number of inner iterations to update the thermostats
 \param skip_restart Skip initialization of the restart information
 */
TwoStepNHRigid::TwoStepNHRigid(boost::shared_ptr<SystemDefinition> sysdef,
                               boost::shared_ptr<ParticleGroup> group,
                               const std::string& suffix,
                               unsigned int tchain,
                               unsigned int pchain,
                               unsigned int iter)
    : IntegrationMethodTwoStep(sysdef, group)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNHRigid" << endl;

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

    m_q_t = m_q_r = m_q_b = NULL;
    m_eta_t = m_eta_r = m_eta_b = NULL;
    m_eta_dot_t = m_eta_dot_r = m_eta_dot_b = NULL;
    m_f_eta_t = m_f_eta_r = m_f_eta_b = NULL;
    m_w = m_wdti1 = m_wdti2 = m_wdti4 = NULL;

    // Using thermostat or barostat
    m_tstat = false;
    m_pstat = false;

    m_boltz = Scalar(1.0);
    m_tchain = tchain;
    m_pchain = pchain;
    m_order = 3;
    m_iter = iter;
    m_couple = couple_xyz;

    m_mtk_term1 = m_mtk_term2 = Scalar(0.0);
    m_akin_t = m_akin_r = Scalar(0.0);
    m_nf_t = m_nf_r = m_g_f = Scalar(0.0);
    }

TwoStepNHRigid::~TwoStepNHRigid()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNHRigid" << endl;
    if (m_tstat)
        deallocate_tchain();
    if (m_pstat)
        deallocate_pchain();

    if (m_tstat || m_pstat)
        {
        delete [] m_w;
        delete [] m_wdti1;
        delete [] m_wdti2;
        delete [] m_wdti4;
        }
    }

/* Setup computes the initial body forces and torques prior to the first update step

*/

void TwoStepNHRigid::setup()
    {
    if (m_prof)
        m_prof->push("Rigid setup");

    // Get the number of rigid bodies for frequent use
    m_n_bodies = m_body_group->getNumMembers();

    // Get the system dimensionality
    m_dimension = m_sysdef->getNDimensions();

    m_dt_half = Scalar(0.5) * m_deltaT;

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
    if (m_dimension == 3)
        {
        m_nf_t = 3 * m_n_bodies;
        m_nf_r = 3 * m_n_bodies;

        //! Subtract from nf_r one for each singular moment inertia of a rigid body
        for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
            {
            unsigned int body = m_body_group->getMemberIndex(group_idx);

            if (fabs(moment_inertia_handle.data[body].x) < EPSILON) m_nf_r -= 1.0;
            if (fabs(moment_inertia_handle.data[body].y) < EPSILON) m_nf_r -= 1.0;
            if (fabs(moment_inertia_handle.data[body].z) < EPSILON) m_nf_r -= 1.0;
            }
        }
    else // m_dimension == 2
        {
        m_nf_t = 2 * m_n_bodies;
        m_nf_r = m_n_bodies;

        //! Subtract from nf_r one for each singular moment inertia of a rigid body
        for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
            {
            unsigned int body = m_body_group->getMemberIndex(group_idx);

            if (fabs(moment_inertia_handle.data[body].z) < EPSILON) m_nf_r -= 1.0;
            }
        }

    m_g_f = m_nf_t + m_nf_r;

    // Reset all forces and torques
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);

        vel_handle.data[body].x = Scalar(0.0);
        vel_handle.data[body].y = Scalar(0.0);
        vel_handle.data[body].z = Scalar(0.0);

        force_handle.data[body].x = Scalar(0.0);
        force_handle.data[body].y = Scalar(0.0);
        force_handle.data[body].z = Scalar(0.0);

        torque_handle.data[body].x = Scalar(0.0);
        torque_handle.data[body].y = Scalar(0.0);
        torque_handle.data[body].z = Scalar(0.0);

        angmom_handle.data[body].x = Scalar(0.0);
        angmom_handle.data[body].y = Scalar(0.0);
        angmom_handle.data[body].z = Scalar(0.0);
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

    m_akin_t = m_akin_r = Scalar(0.0);
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);

        vel_handle.data[body].x /= body_mass_handle.data[body];
        vel_handle.data[body].y /= body_mass_handle.data[body];
        vel_handle.data[body].z /= body_mass_handle.data[body];

        computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body],
                               ex_space_handle.data[body], ey_space_handle.data[body],
                               ez_space_handle.data[body], angvel_handle.data[body]);

        m_akin_t += body_mass_handle.data[body] * (vel_handle.data[body].x * vel_handle.data[body].x +
                                                   vel_handle.data[body].y * vel_handle.data[body].y +
                                                   vel_handle.data[body].z * vel_handle.data[body].z);
        m_akin_r += angmom_handle.data[body].x * angvel_handle.data[body].x
                  + angmom_handle.data[body].y * angvel_handle.data[body].y
                  + angmom_handle.data[body].z * angvel_handle.data[body].z;
        }

    Scalar4 mbody;
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);

        matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body], angmom_handle.data[body], mbody);
        quatvec(orientation_handle.data[body], mbody, conjqm_handle.data[body]);

        conjqm_handle.data[body].x *= Scalar(2.0);
        conjqm_handle.data[body].y *= Scalar(2.0);
        conjqm_handle.data[body].z *= Scalar(2.0);
        conjqm_handle.data[body].w *= Scalar(2.0);
        }

    } // out of scope for handles

    if (m_tstat)
        allocate_tchain();

    if (m_pstat)
        {
        allocate_pchain();

        // determine the number of directions barostatted
        m_pdim = 0;
        if (m_flags & baro_x)
            m_pdim++;
        if (m_flags & baro_y)
            m_pdim++;
        if (m_flags & baro_z)
            m_pdim++;
        }

    if (m_tstat || m_pstat)
        {
        m_w     = new Scalar [m_order];
        m_wdti1 = new Scalar [m_order];
        m_wdti2 = new Scalar [m_order];
        m_wdti4 = new Scalar [m_order];

        if (m_order == 3)
            {
            m_w[0] = 1.0 / (2.0 - pow(2.0, 1.0/3.0));
            m_w[1] = 1.0 - 2.0*m_w[0];
            m_w[2] = m_w[0];
            }
        else if (m_order == 5)
            {
            m_w[0] = 1.0 / (4.0 - pow(4.0, 1.0/3.0));
            m_w[1] = m_w[0];
            m_w[2] = 1.0 - 4.0*m_w[0];
            m_w[3] = m_w[0];
            m_w[4] = m_w[0];
            }

        // update order/timestep-dependent coefficients

        for (unsigned int i = 0; i < m_order; i++)
            {
            m_wdti1[i] = m_w[i] * m_deltaT / m_iter;
            m_wdti2[i] = m_wdti1[i] / 2.0;
            m_wdti4[i] = m_wdti1[i] / 4.0;
            }
        }

    // computes the total number of degrees of freedom used for system temperature compute
    ArrayHandle< unsigned int > h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

    unsigned int non_rigid_count = 0;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        if (h_body.data[i] == NO_BODY) non_rigid_count++;

    unsigned int rigid_dof = m_sysdef->getRigidData()->getNumDOF();
    m_dof = m_dimension * non_rigid_count + rigid_dof;

    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepNHRigid::integrateStepOne(unsigned int timestep)
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
        m_prof->push("NH rigid step 1");

    // get box
    const BoxDim& box = m_pdata->getBox();

    Scalar tmp, scale_r;
    Scalar3 scale_t, scale_v;
    scale_t = make_scalar3(Scalar(1.0), Scalar(1.0), Scalar(1.0));
    scale_v = make_scalar3(Scalar(1.0), Scalar(1.0), Scalar(1.0));
    scale_r = Scalar(1.0);

    // velocity scaling factors for translation and rotation
    if (m_tstat)
        {
        // velocity scaling factors from thermostat chain
        scale_t.x = scale_t.y = scale_t.z = fast::exp(-m_dt_half * m_eta_dot_t[0]);
        scale_r = fast::exp(-m_dt_half * m_eta_dot_r[0]);
        }

    if (m_pstat)
        {
        // velocity scaling factors from thermostat chain coupled with barostat
        scale_t.x *= fast::exp(-m_dt_half * (m_epsilon_dot[0] + m_mtk_term2));
        scale_t.y *= fast::exp(-m_dt_half * (m_epsilon_dot[3] + m_mtk_term2));
        scale_t.z *= fast::exp(-m_dt_half * (m_epsilon_dot[5] + m_mtk_term2));
        scale_r *= fast::exp(-m_dt_half * m_pdim * m_mtk_term2);

        tmp = m_dt_half * m_epsilon_dot[0];
        scale_v.x = m_deltaT * fast::exp(tmp) * maclaurin_series(tmp);
        tmp = m_dt_half * m_epsilon_dot[3];
        scale_v.y = m_deltaT * fast::exp(tmp) * maclaurin_series(tmp);
        tmp = m_dt_half * m_epsilon_dot[5];
        scale_v.z = m_deltaT * fast::exp(tmp) * maclaurin_series(tmp);
        }

    if (m_tstat || m_pstat)
        m_akin_t = m_akin_r = Scalar(0.0);

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
    ArrayHandle<Scalar4> conjqm_handle(m_rigid_data->getConjqm(), access_location::host, access_mode::readwrite);

    Scalar4 mbody, tbody, fquat;
    Scalar dtfm;

    // for each body
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);

        dtfm = m_dt_half / body_mass_handle.data[body];
        vel_handle.data[body].x += dtfm * force_handle.data[body].x;
        vel_handle.data[body].y += dtfm * force_handle.data[body].y;
        vel_handle.data[body].z += dtfm * force_handle.data[body].z;

        if (m_tstat || m_pstat)
            {
            vel_handle.data[body].x *= scale_t.x;
            vel_handle.data[body].y *= scale_t.y;
            vel_handle.data[body].z *= scale_t.z;
            tmp = vel_handle.data[body].x * vel_handle.data[body].x + vel_handle.data[body].y * vel_handle.data[body].y +
                  vel_handle.data[body].z * vel_handle.data[body].z;
            m_akin_t += body_mass_handle.data[body] * tmp;
            }

        // step 1.2 - update xcm by full step
        if (!m_pstat)
            {
            com_handle.data[body].x += vel_handle.data[body].x * m_deltaT;
            com_handle.data[body].y += vel_handle.data[body].y * m_deltaT;
            com_handle.data[body].z += vel_handle.data[body].z * m_deltaT;
            }
        else
            {
            com_handle.data[body].x += scale_v.x * vel_handle.data[body].x;
            com_handle.data[body].y += scale_v.y * vel_handle.data[body].y;
            com_handle.data[body].z += scale_v.z * vel_handle.data[body].z;
            }

        box.wrap(com_handle.data[body], body_image_handle.data[body]);

        if (m_tstat || m_pstat)
            {
            matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body],
                       ez_space_handle.data[body], torque_handle.data[body], tbody);
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

            no_squish_rotate(3, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], m_dt_half);
            no_squish_rotate(2, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], m_dt_half);
            no_squish_rotate(1, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], m_deltaT);
            no_squish_rotate(2, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], m_dt_half);
            no_squish_rotate(3, conjqm_handle.data[body], orientation_handle.data[body], moment_inertia_handle.data[body], m_dt_half);

            // update the exyz_space
            // transform p back to angmom
            // update angular velocity

            exyzFromQuaternion(orientation_handle.data[body], ex_space_handle.data[body],
                               ey_space_handle.data[body], ez_space_handle.data[body]);
            invquatvec(orientation_handle.data[body], conjqm_handle.data[body], mbody);
            transpose_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body],
                          mbody, angmom_handle.data[body]);

            angmom_handle.data[body].x *= Scalar(0.5);
            angmom_handle.data[body].y *= Scalar(0.5);
            angmom_handle.data[body].z *= Scalar(0.5);

            computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body],
                                   ex_space_handle.data[body], ey_space_handle.data[body],
                                   ez_space_handle.data[body], angvel_handle.data[body]);

            m_akin_r += angmom_handle.data[body].x * angvel_handle.data[body].x
                    + angmom_handle.data[body].y * angvel_handle.data[body].y
                    + angmom_handle.data[body].z * angvel_handle.data[body].z;
            }
        else
            {
            // update the angular momentum
            angmom_handle.data[body].x += m_dt_half * torque_handle.data[body].x;
            angmom_handle.data[body].y += m_dt_half * torque_handle.data[body].y;
            angmom_handle.data[body].z += m_dt_half * torque_handle.data[body].z;

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
        }


    } // out of scope for handles

    if (m_pstat)
        {
        // rescale box dimensions and remap body COM's
        remap();
        }

    if (m_tstat)
        {
        // compute thermostat chain coupled with thermostat
        update_nh_tchain(m_akin_t, m_akin_r, timestep);
        }

    if (m_pstat)
        {
        // compute target pressure
        compute_target_pressure(timestep);

        // compute thermostat chain coupled with barotat
        update_nh_pchain(timestep);
        }

    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNHRigid::integrateStepTwo(unsigned int timestep)
    {
    // sanity check
    if (m_n_bodies <= 0)
        return;

    // compute net forces and torques on rigid bodies from particle forces
    computeForceAndTorque(timestep);

    if (m_prof)
        m_prof->push("NH rigid step 2");

    Scalar scale_r;
    Scalar3 scale_t;
    scale_t = make_scalar3(Scalar(1.0), Scalar(1.0), Scalar(1.0));
    scale_r = Scalar(1.0);

    // velocity scaling factors from the thermostat chain
    if (m_tstat)
        {
        scale_t.x = scale_t.y = scale_t.z = fast::exp(-m_dt_half * m_eta_dot_t[0]);
        scale_r = fast::exp(-m_dt_half * m_eta_dot_r[0]);
        }
    // and from the thermostat chain coupled with barostat
    if (m_pstat)
        {
        scale_t.x *= fast::exp(-m_dt_half * (m_epsilon_dot[0] + m_mtk_term2));
        scale_t.y *= fast::exp(-m_dt_half * (m_epsilon_dot[3] + m_mtk_term2));
        scale_t.z *= fast::exp(-m_dt_half * (m_epsilon_dot[5] + m_mtk_term2));
        scale_r *= fast::exp(-m_dt_half * m_pdim * m_mtk_term2);
        m_akin_t = m_akin_r = Scalar(0.0);
        }

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
    ArrayHandle<Scalar4> conjqm_handle(m_rigid_data->getConjqm(), access_location::host, access_mode::readwrite);

    Scalar4 mbody, tbody, fquat;
    Scalar dtfm;

    // 2nd step: final integration
    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);
        dtfm = m_dt_half / body_mass_handle.data[body];

        if (m_tstat || m_pstat)
            {
            vel_handle.data[body].x = scale_t.x * vel_handle.data[body].x + dtfm * force_handle.data[body].x;
            vel_handle.data[body].y = scale_t.y * vel_handle.data[body].y + dtfm * force_handle.data[body].y;
            vel_handle.data[body].z = scale_t.z * vel_handle.data[body].z + dtfm * force_handle.data[body].z;

            // update conjqm, then transform to angmom, set velocity again
            matrix_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body],
                       torque_handle.data[body], tbody);
            quatvec(orientation_handle.data[body], tbody, fquat);

            conjqm_handle.data[body].x = scale_r * conjqm_handle.data[body].x + m_deltaT * fquat.x;
            conjqm_handle.data[body].y = scale_r * conjqm_handle.data[body].y + m_deltaT * fquat.y;
            conjqm_handle.data[body].z = scale_r * conjqm_handle.data[body].z + m_deltaT * fquat.z;
            conjqm_handle.data[body].w = scale_r * conjqm_handle.data[body].w + m_deltaT * fquat.w;

            invquatvec(orientation_handle.data[body], conjqm_handle.data[body], mbody);
            transpose_dot(ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body],
                          mbody, angmom_handle.data[body]);

            angmom_handle.data[body].x *= Scalar(0.5);
            angmom_handle.data[body].y *= Scalar(0.5);
            angmom_handle.data[body].z *= Scalar(0.5);

            computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body],
                                   ex_space_handle.data[body],
                                   ey_space_handle.data[body],
                                   ez_space_handle.data[body],
                                   angvel_handle.data[body]);
            if (m_pstat)
                {
                m_akin_t += body_mass_handle.data[body] * (vel_handle.data[body].x * vel_handle.data[body].x +
                                                           vel_handle.data[body].y * vel_handle.data[body].y +
                                                           vel_handle.data[body].z * vel_handle.data[body].z);
                m_akin_r += angmom_handle.data[body].x * angvel_handle.data[body].x +
                            angmom_handle.data[body].y * angvel_handle.data[body].y +
                            angmom_handle.data[body].z * angvel_handle.data[body].z;
                }
            }
        else
            {
            vel_handle.data[body].x += dtfm * force_handle.data[body].x;
            vel_handle.data[body].y += dtfm * force_handle.data[body].y;
            vel_handle.data[body].z += dtfm * force_handle.data[body].z;

            angmom_handle.data[body].x += m_dt_half * torque_handle.data[body].x;
            angmom_handle.data[body].y += m_dt_half * torque_handle.data[body].y;
            angmom_handle.data[body].z += m_dt_half * torque_handle.data[body].z;

            computeAngularVelocity(angmom_handle.data[body], moment_inertia_handle.data[body],
                                   ex_space_handle.data[body], ey_space_handle.data[body], ez_space_handle.data[body],
                                   angvel_handle.data[body]);
            }
        }
    } // out of scope for handles

    if (m_pstat)
        {
        // compute current pressure
        compute_current_pressure(timestep);

        // compute target pressure
        compute_target_pressure(timestep);

        // compute barostat
        update_nh_barostat(m_akin_t, m_akin_r);
        }

    if (m_prof)
        m_prof->pop();
    }

/*! \param query_group Group over which to count degrees of freedom.
    A majority of the integration methods add D degrees of freedom per particle in \a query_group that is also in the
    group assigned to the method. Hence, the base class IntegrationMethodTwoStep will implement that counting.
    Derived classes can ovveride if needed.
*/
unsigned int TwoStepNHRigid::getNDOF(boost::shared_ptr<ParticleGroup> query_group)
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
void TwoStepNHRigid::computeForceAndTorque(unsigned int timestep)
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

        force_handle.data[body].x = Scalar(0.0);
        force_handle.data[body].y = Scalar(0.0);
        force_handle.data[body].z = Scalar(0.0);

        torque_handle.data[body].x = Scalar(0.0);
        torque_handle.data[body].y = Scalar(0.0);
        torque_handle.data[body].z = Scalar(0.0);
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

    TwoStepNHRigid acts as the base class for all rigid body integration methods. Check here that all particles belong
    to rigid bodies.
*/
void TwoStepNHRigid::validateGroup()
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
void TwoStepNHRigid::update_nh_tchain(Scalar akin_t, Scalar akin_r, unsigned int timestep)
    {
    unsigned int i, j, k;
    Scalar kt, gfkt_t, gfkt_r, tmp, ms, s, s2;

    Scalar t_target = m_temperature->getValue(timestep);
    kt = m_boltz * t_target;
    gfkt_t = m_nf_t * kt;
    gfkt_r = m_nf_r * kt;

    // update thermostat masses

    Scalar t_mass = kt / (m_tfreq * m_tfreq);
    m_q_t[0] = m_nf_t * t_mass;
    m_q_r[0] = m_nf_r * t_mass;
    for (i = 1; i < m_tchain; i++)
        m_q_t[i] = m_q_r[i] = t_mass;

    // update force of thermostats coupled to particles

    m_f_eta_t[0] = (akin_t - gfkt_t) / m_q_t[0];
    m_f_eta_r[0] = (akin_r - gfkt_r) / m_q_r[0];

    // multiple timestep iteration

    for (i = 0; i < m_iter; i++)
        {
        for (j = 0; j < m_order; j++)
            {

            // update thermostat velocities half step

            m_eta_dot_t[m_tchain-1] += m_wdti2[j] * m_f_eta_t[m_tchain-1];
            m_eta_dot_r[m_tchain-1] += m_wdti2[j] * m_f_eta_r[m_tchain-1];

            for (k = 1; k < m_tchain; k++)
                {
                tmp = m_wdti4[j] * m_eta_dot_t[m_tchain-k];
                ms = maclaurin_series(tmp);
                s = fast::exp(-1.0 * tmp);
                s2 = s * s;
                m_eta_dot_t[m_tchain-k-1] = m_eta_dot_t[m_tchain-k-1] * s2 +
                                       m_wdti2[j] * m_f_eta_t[m_tchain-k-1] * s * ms;

                tmp = m_wdti4[j] * m_eta_dot_r[m_tchain-k];
                ms = maclaurin_series(tmp);
                s = fast::exp(-1.0 * tmp);
                s2 = s * s;
                m_eta_dot_r[m_tchain-k-1] = m_eta_dot_r[m_tchain-k-1] * s2 +
                                       m_wdti2[j] * m_f_eta_r[m_tchain-k-1] * s * ms;
                }

            // update thermostat positions a full step

            for (k = 0; k < m_tchain; k++)
                {
                m_eta_t[k] += m_wdti1[j] * m_eta_dot_t[k];
                m_eta_r[k] += m_wdti1[j] * m_eta_dot_r[k];
                }

            // update thermostat forces

            for (k = 1; k < m_tchain; k++)
                {
                m_f_eta_t[k] = m_q_t[k-1] * m_eta_dot_t[k-1] * m_eta_dot_t[k-1] - kt;
                m_f_eta_t[k] /= m_q_t[k];
                m_f_eta_r[k] = m_q_r[k-1] * m_eta_dot_r[k-1] * m_eta_dot_r[k-1] - kt;
                m_f_eta_r[k] /= m_q_r[k];
                }

            // update thermostat velocities a full step

            for (k = 0; k < m_tchain-1; k++)
                {
                tmp = m_wdti4[j] * m_eta_dot_t[k+1];
                ms = maclaurin_series(tmp);
                s = fast::exp(-1.0 * tmp);
                s2 = s * s;
                m_eta_dot_t[k] = m_eta_dot_t[k] * s2 + m_wdti2[j] * m_f_eta_t[k] * s * ms;
                tmp = m_q_t[k] * m_eta_dot_t[k] * m_eta_dot_t[k] - kt;
                m_f_eta_t[k+1] = tmp / m_q_t[k+1];

                tmp = m_wdti4[j] * m_eta_dot_r[k+1];
                ms = maclaurin_series(tmp);
                s = fast::exp(-1.0 * tmp);
                s2 = s * s;
                m_eta_dot_r[k] = m_eta_dot_r[k] * s2 + m_wdti2[j] * m_f_eta_r[k] * s * ms;
                tmp = m_q_r[k] * m_eta_dot_r[k] * m_eta_dot_r[k] - kt;
                m_f_eta_r[k+1] = tmp / m_q_r[k+1];
                }

            m_eta_dot_t[m_tchain-1] += m_wdti2[j] * m_f_eta_t[m_tchain-1];
            m_eta_dot_r[m_tchain-1] += m_wdti2[j] * m_f_eta_r[m_tchain-1];
            }
        }

    }

/*! Update Nose-Hoover thermostats coupled with barostat
    \param timestep Current time step
*/
void TwoStepNHRigid::update_nh_pchain(unsigned int timestep)
    {
    unsigned int i, j, k;
    Scalar kt, tmp, ms, s, s2;

    if (m_tstat) kt = m_boltz * m_temperature->getValue(timestep);
    else kt = m_boltz;

    // update forces acting on thermostat

    Scalar tb_mass = kt / (m_pfreq * m_pfreq);
    m_q_b[0] = m_dimension * m_dimension * tb_mass;
    for (k = 1; k < m_pchain; k++)
        {
        m_q_b[k] = tb_mass;
        m_f_eta_b[k] = m_q_b[k-1] * m_eta_dot_b[k-1] * m_eta_dot_b[k-1] - kt;
        m_f_eta_b[k] /= m_q_b[k];
        }

    // update thermostat masses

    Scalar kecurrent = Scalar(0.0);
    if (m_flags & baro_x)
        {
        m_epsilon_mass[0] = (m_g_f + m_dimension) * kt / (m_pfreq * m_pfreq);
        kecurrent += m_epsilon_mass[0] * m_epsilon_dot[0] * m_epsilon_dot[0];
        }

    if (m_flags & baro_y)
        {
        m_epsilon_mass[3] = (m_g_f + m_dimension) * kt / (m_pfreq * m_pfreq);
        kecurrent += m_epsilon_mass[3] * m_epsilon_dot[3] * m_epsilon_dot[3];
        }

    if (m_flags & baro_z)
        {
        m_epsilon_mass[5] = (m_g_f + m_dimension) * kt / (m_pfreq * m_pfreq);
        kecurrent += m_epsilon_mass[5] * m_epsilon_dot[5] * m_epsilon_dot[5];
        }
    kecurrent /= m_pdim;

    m_f_eta_b[0] = (kecurrent - kt) / m_q_b[0];

    // multiple timestep iteration

    for (i = 0; i < m_iter; i++)
        {
        for (j = 0; j < m_order; j++)
            {
            // update thermostat velocities a half step
            m_eta_dot_b[m_pchain-1] += m_wdti2[j] * m_f_eta_b[m_pchain-1];

            for (k = 1; k < m_pchain; k++)
                {
                tmp = m_wdti4[j] * m_eta_dot_b[m_pchain-k];
                ms = maclaurin_series(tmp);
                s = fast::exp(-0.5 * tmp);
                s2 = s * s;
                m_eta_dot_b[m_pchain-k-1] = m_eta_dot_b[m_pchain-k-1] * s2 +
                  m_wdti2[j] * m_f_eta_b[m_pchain-k-1] * s * ms;
                }

            // update thermostat positions
            for (k = 0; k < m_pchain; k++)
                m_eta_b[k] += m_wdti1[j] * m_eta_dot_b[k];

            // update thermostat forces
            for (k = 1; k < m_pchain; k++)
                {
                m_f_eta_b[k] = m_q_b[k-1] * m_eta_dot_b[k-1] * m_eta_dot_b[k-1] - kt;
                m_f_eta_b[k] /= m_q_b[k];
                }

            // update thermostat velocites a full step
            for (k = 0; k < m_pchain-1; k++)
                {
                tmp = m_wdti4[j] * m_eta_dot_b[k+1];
                ms = maclaurin_series(tmp);
                s = fast::exp(-0.5 * tmp);
                s2 = s * s;
                m_eta_dot_b[k] = m_eta_dot_b[k] * s2 + m_wdti2[j] * m_f_eta_b[k] * s * ms;
                m_f_eta_b[k+1] = (m_q_b[k] * m_eta_dot_b[k] * m_eta_dot_b[k] - kt) / m_q_b[k+1];
                }

            m_eta_dot_b[m_pchain-1] += m_wdti2[j] * m_f_eta_b[m_pchain-1];
            }
        }
    }


/*! Update the volume scaling factor
  \param akin_t Translational kinetic energy
  \param akin_r Rotational kinetic energy
*/
void TwoStepNHRigid::update_nh_barostat(Scalar akin_t, Scalar akin_r)
    {
    Scalar volume, scale, f_epsilon;

    // get box
    BoxDim box = m_pdata->getBox();
    Scalar3 L = box.getL();

    if (m_dimension == 2)
        volume = L.x * L.y;
    else
        volume = L.x * L.y * L.z;

    m_mtk_term1 = (akin_t + akin_r) / m_g_f;

    m_mtk_term2 = Scalar(0.0);
    scale = fast::exp(-m_dt_half * m_eta_dot_b[0]);
    if (m_flags & baro_x)
        {
        f_epsilon = ((m_curr_P[0] - m_p_hydro) * volume + m_mtk_term1) / m_epsilon_mass[0];
        m_epsilon_dot[0] += m_dt_half * f_epsilon;
        m_epsilon_dot[0] *= scale;
        m_mtk_term2 += m_epsilon_dot[0];
        }

    if (m_flags & baro_y)
        {
        f_epsilon = ((m_curr_P[3] - m_p_hydro) * volume + m_mtk_term1) / m_epsilon_mass[3];
        m_epsilon_dot[3] += m_dt_half * f_epsilon;
        m_epsilon_dot[3] *= scale;
        m_mtk_term2 += m_epsilon_dot[3];
        }

    if (m_flags & baro_z)
        {
        f_epsilon = ((m_curr_P[5] - m_p_hydro) * volume + m_mtk_term1) / m_epsilon_mass[5];
        m_epsilon_dot[5] += m_dt_half * f_epsilon;
        m_epsilon_dot[5] *= scale;
        m_mtk_term2 += m_epsilon_dot[5];
        }

    m_mtk_term2 /= m_g_f;

    if (m_flags & baro_xy)
        {
        f_epsilon = m_curr_P[1] * volume / m_epsilon_mass[1];
        m_epsilon_dot[1] += m_dt_half * f_epsilon;
        m_epsilon_dot[1] *= scale;
        }

    if (m_flags & baro_xz)
        {
        f_epsilon = m_curr_P[2] * volume / m_epsilon_mass[2];
        m_epsilon_dot[2] += m_dt_half * f_epsilon;
        m_epsilon_dot[2] *= scale;
        }

    if (m_flags & baro_yz)
        {
        f_epsilon = m_curr_P[4] * volume / m_epsilon_mass[4];
        m_epsilon_dot[4] += m_dt_half * f_epsilon;
        m_epsilon_dot[4] *= scale;
        }
    }

/*! Compute current pressure
  \param timestep Current timestep
*/

void TwoStepNHRigid::compute_current_pressure(unsigned int timestep)
{
    // compute the current thermodynamic properties- note that compute might not be available at the current timestep
    m_thermo_group->compute(timestep);

    // compute pressure for the next half time step
    PressureTensor P = m_thermo_group->getPressureTensor();

    if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
        {
        Scalar extP = m_pressure->getValue(timestep);
        P.xx = P.yy = P.zz = extP;
        P.xy = P.xz = P.yz = Scalar(0.0);
        }

    if (m_couple == couple_xyz)
        {
        m_curr_P[0] = m_curr_P[3] = m_curr_P[5] = Scalar(1.0/3.0) * (P.xx + P.yy + P.zz);
        }
    else if (m_couple == couple_xy)
        {
        m_curr_P[0] = m_curr_P[3] = Scalar(0.5) * (P.xx + P.yy);
        m_curr_P[5] = P.zz;
        }
    else if (m_couple == couple_yz)
        {
        m_curr_P[3] = m_curr_P[5] = Scalar(0.5) * (P.yy + P.zz);
        m_curr_P[0] = P.xx;
        }
    else if (m_couple == couple_xz)
        {
        m_curr_P[0] = m_curr_P[5] = Scalar(0.5) * (P.xx + P.zz);
        m_curr_P[3] = P.yy;
        }
    else
        {
        m_curr_P[0] = P.xx;
        m_curr_P[3] = P.yy;
        m_curr_P[5] = P.zz;
        }

    // switch order from xy-xz-yz to Voigt

    if (m_flags & baro_xy || m_flags & baro_xz || m_flags & baro_yz)
        {
        m_curr_P[1] = P.yz;
        m_curr_P[2] = P.xz;
        m_curr_P[4] = P.xy;
        }
}

/*! Compute target pressure
  \param timestep Current timestep
*/
void TwoStepNHRigid::compute_target_pressure(unsigned int timestep)
{
    Scalar p_target = m_pressure->getValue(timestep);

    m_p_hydro = Scalar(0.0);
    if (m_flags & baro_x)
        {
        m_target_P[0] = p_target;
        m_p_hydro += p_target;
        }
    if (m_flags & baro_y)
        {
        m_target_P[3] = p_target;
        m_p_hydro += p_target;
        }
    if (m_flags & baro_z)
        {
        m_target_P[5] = p_target;
        m_p_hydro += p_target;
        }
    m_p_hydro /= m_pdim;

    if (m_flags & baro_xy)
        m_target_P[1] = p_target;
    if (m_flags & baro_xz)
        m_target_P[2] = p_target;
    if (m_flags & baro_yz)
        m_target_P[4] = p_target;
}


/*! Calculate the new box size from dilation
    Remap the rigid body COMs from old box to new box
    Note that NPT rigid currently only deals with rigid bodies, no point particles
    For hybrid systems, use TwoStepNPT coupled with TwoStepNVTRigid to avoid duplicating box resize
*/
void TwoStepNHRigid::remap()
    {
    BoxDim curBox = m_pdata->getGlobalBox();
    Scalar3 curL = curBox.getL();
    Scalar3 L = curL;

    if (m_flags & baro_x)
        L.x = curL.x * fast::exp(m_dt_half * m_epsilon_dot[0]);
    if (m_flags & baro_y)
        L.y = curL.y * fast::exp(m_dt_half * m_epsilon_dot[3]);
    if (m_flags & baro_z)
        L.z = curL.z * fast::exp(m_dt_half * m_epsilon_dot[5]);

    // copy and setL
    BoxDim newBox = curBox;
    newBox.setL(L);

    // set the new box
    m_pdata->setGlobalBox(newBox);

    // convert rigid body COMs to lamda coords

    ArrayHandle<Scalar4> com_handle(m_rigid_data->getCOM(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < m_n_bodies; group_idx++)
        {
        unsigned int body = m_body_group->getMemberIndex(group_idx);

        Scalar3 f = curBox.makeFraction(make_scalar3(com_handle.data[body].x,
                                                     com_handle.data[body].y,
                                                     com_handle.data[body].z));
        Scalar3 scaled_cm = newBox.makeCoordinates(f);
        com_handle.data[body].x = scaled_cm.x;
        com_handle.data[body].y = scaled_cm.y;
        com_handle.data[body].z = scaled_cm.z;
        }

    }

/*! Allocate memory for thermostat chains

*/
void TwoStepNHRigid::allocate_tchain()
    {
    m_q_t = new Scalar [m_tchain];
    m_q_r = new Scalar [m_tchain];
    m_eta_t = new Scalar [m_tchain];
    m_eta_r = new Scalar [m_tchain];
    m_eta_dot_t = new Scalar [m_tchain];
    m_eta_dot_r = new Scalar [m_tchain];
    m_f_eta_t = new Scalar [m_tchain];
    m_f_eta_r = new Scalar [m_tchain];

    for (unsigned int i = 0; i < m_tchain; i++)
        {
        m_eta_t[i] = m_eta_dot_t[i] = m_f_eta_t[i] = Scalar(0.0);
        m_eta_r[i] = m_eta_dot_r[i] = m_f_eta_r[i] = Scalar(0.0);
        }
    }

/*! Allocate memory for thermostat chains coupled with barostat

*/
void TwoStepNHRigid::allocate_pchain()
    {
    m_q_b = new Scalar [m_pchain];
    m_eta_b = new Scalar [m_pchain];
    m_eta_dot_b = new Scalar [m_pchain];
    m_f_eta_b = new Scalar [m_pchain];

    for (unsigned int i = 0; i < m_pchain; i++)
        m_eta_b[i] = m_eta_dot_b[i] = m_f_eta_b[i] = Scalar(0.0);
    }

/*! Deallocate memory for thermostat chains

*/
void TwoStepNHRigid::deallocate_tchain()
    {
    delete [] m_q_t;
    delete [] m_q_r;
    delete [] m_eta_t;
    delete [] m_eta_r;
    delete [] m_eta_dot_t;
    delete [] m_eta_dot_r;
    delete [] m_f_eta_t;
    delete [] m_f_eta_r;
    }

/*! Deallocate memory for thermostat chains coupled with barostat

*/
void TwoStepNHRigid::deallocate_pchain()
    {
    delete [] m_q_b;
    delete [] m_eta_b;
    delete [] m_eta_dot_b;
    delete [] m_f_eta_b;
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepNHRigid::getProvidedLogQuantities()
    {
    return m_log_names;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quanity logged here
*/

Scalar TwoStepNHRigid::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    return Scalar(0);
    }

void export_TwoStepNHRigid()
    {
    scope in_nh_rigid = class_<TwoStepNHRigid, boost::shared_ptr<TwoStepNHRigid>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
        ("TwoStepNHRigid", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       const std::string&,
                       unsigned int,
                       unsigned int,
                       unsigned int >())
        .def("setT", &TwoStepNHRigid::setT)
        .def("setP", &TwoStepNHRigid::setP)
        .def("setTau", &TwoStepNHRigid::setTau)
        .def("setTauP", &TwoStepNHRigid::setTauP)
        .def("setPartialScale", &TwoStepNHRigid::setPartialScale)
        ;

    enum_<TwoStepNHRigid::couplingMode>("couplingMode")
    .value("couple_none", TwoStepNHRigid::couple_none)
    .value("couple_xy", TwoStepNHRigid::couple_xy)
    .value("couple_xz", TwoStepNHRigid::couple_xz)
    .value("couple_yz", TwoStepNHRigid::couple_yz)
    .value("couple_xyz", TwoStepNHRigid::couple_xyz)
    ;

    enum_<TwoStepNHRigid::baroFlags>("baroFlags")
    .value("baro_x", TwoStepNHRigid::baro_x)
    .value("baro_y", TwoStepNHRigid::baro_y)
    .value("baro_z", TwoStepNHRigid::baro_z)
    .value("baro_xy", TwoStepNHRigid::baro_xy)
    .value("baro_xz", TwoStepNHRigid::baro_xz)
    .value("baro_yz", TwoStepNHRigid::baro_yz)
    ;
    }

