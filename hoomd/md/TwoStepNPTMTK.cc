// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepNPTMTK.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

using namespace std;

/*! \file TwoStepNPTMTK.cc
    \brief Contains code for the TwoStepNPTMTK class
*/

namespace hoomd
    {
namespace md
    {
//! Coefficients of f(x) = sinh(x)/x = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 * x^8 + a_10 *
//! x^10
const Scalar f_coeff[] = {Scalar(1.0),
                          Scalar(1.0 / 6.0),
                          Scalar(1.0 / 120.0),
                          Scalar(1.0 / 5040.0),
                          Scalar(1.0 / 362880.0),
                          Scalar(1.0 / 39916800.0)};

//! Coefficients of g(x) = coth(x) - 1/x =  a_1 * x + a_3 * x^3 + a_5 * x^5 + a_7 * x^7 + a_9 * x^9
const Scalar g_coeff[] = {Scalar(1.0 / 3.0),
                          Scalar(-1.0 / 45.0),
                          Scalar(2.0 / 945.0),
                          Scalar(-1.0 / 4725.0),
                          Scalar(1.0 / 93555.0)};

//! Coefficients of h(x) = (-1/sinh^2(x)+1/x^2) = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 *
//! x^8 + a_10 * x^10
const Scalar h_coeff[] = {Scalar(1.0 / 3.0),
                          Scalar(-1.0 / 15.0),
                          Scalar(2.0 / 189.0),
                          Scalar(-1.0 / 675.0),
                          Scalar(2.0 / 10395.0),
                          Scalar(-1382.0 / 58046625.0)};

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_half_step Own ComputeThermo to compute thermo properties of the integrated \a
   group (at half time step) \param thermo_half_step_full_tstep ComputeThermo to compute thermo
   properties of the integrated \a group at full time step \param tau NPT temperature period \param
   tauS NPT pressure period \param T Temperature set point \param S Pressure or Stress set point.
   Pressure if one value, Stress if a list of 6. Stress should be ordered as [xx, yy, zz, yz, xz,
   xy] \param couple Coupling mode \param flags Barostatted simulation box degrees of freedom
*/
TwoStepNPTMTK::TwoStepNPTMTK(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             std::shared_ptr<ComputeThermo> thermo_half_step,
                             std::shared_ptr<ComputeThermo> thermo_full_step,
                             Scalar tau,
                             Scalar tauS,
                             std::shared_ptr<Variant> T,
                             const std::vector<std::shared_ptr<Variant>>& S,
                             const std::string& couple,
                             const std::vector<bool>& flags,
                             const bool nph)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo_half_step(thermo_half_step),
      m_thermo_full_step(thermo_full_step), m_ndof(0), m_tau(tau), m_tauS(tauS), m_T(T), m_S(S),
      m_nph(nph), m_rescale_all(false), m_gamma(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNPTMTK" << endl;

    setCouple(couple);
    setFlags(flags);

    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.npt: tau set less than 0.0" << endl;
    if (m_tauS <= 0.0)
        m_exec_conf->msg->warning() << "integrate.npt: tauS set less than 0.0" << endl;

    if (m_flags == 0)
        m_exec_conf->msg->warning() << "integrate.npt: No barostat couplings specified." << endl;

    bool is_two_dimensions = m_sysdef->getNDimensions() == 2;
    m_V = m_pdata->getGlobalBox().getVolume(is_two_dimensions); // volume

    // set initial state
    if (!restartInfoTestValid(getIntegratorVariables(), "npt_mtk", 10))
        {
        initializeIntegratorVariables();
        setValidRestart(false);
        }
    else
        {
        setValidRestart(true);
        }
    }

TwoStepNPTMTK::~TwoStepNPTMTK()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPTMTK" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   Martyna-Tobias-Klein barostat and thermostat
*/
void TwoStepNPTMTK::integrateStepOne(uint64_t timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        m_exec_conf->msg->error() << "integrate.npt(): Integration group empty." << std::endl;
        throw std::runtime_error("Error during NPT integration.");
        }

    // update box dimensions
    bool is_two_dimensions = m_sysdef->getNDimensions() == 2;

    m_V = m_pdata->getGlobalBox().getVolume(is_two_dimensions); // current volume

    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("NPT step 1");

    // update degrees of freedom for MTK term
    m_ndof = m_group->getTranslationalDOF();

    // advance barostat (nuxx, nuyy, nuzz) half a time step
    advanceBarostat(timestep);

    IntegratorVariables v = getIntegratorVariables();
    Scalar nuxx = v.variable[2]; // Barostat tensor, xx component
    Scalar nuxy = v.variable[3]; // Barostat tensor, xy component
    Scalar nuxz = v.variable[4]; // Barostat tensor, xz component
    Scalar nuyy = v.variable[5]; // Barostat tensor, yy component
    Scalar nuyz = v.variable[6]; // Barostat tensor, yz component
    Scalar nuzz = v.variable[7]; // Barostat tensor, zz component

    // Martyna-Tobias-Klein correction
    Scalar mtk = (nuxx + nuyy + nuzz) / (Scalar)m_ndof;

    // update the propagator matrix using current barostat momenta
    updatePropagator(nuxx, nuxy, nuxz, nuyy, nuyz, nuzz);

    // advance box lengths
    BoxDim global_box = m_pdata->getGlobalBox();
    Scalar3 a = global_box.getLatticeVector(0);
    Scalar3 b = global_box.getLatticeVector(1);
    Scalar3 c = global_box.getLatticeVector(2);

    // (a,b,c) are the columns of the (upper triangular) cell parameter matrix
    // multiply with upper triangular matrix
    a.x = m_mat_exp_r[0] * a.x;
    b.x = m_mat_exp_r[0] * b.x + m_mat_exp_r[1] * b.y;
    b.y = m_mat_exp_r[3] * b.y;
    c.x = m_mat_exp_r[0] * c.x + m_mat_exp_r[1] * c.y + m_mat_exp_r[2] * c.z;
    c.y = m_mat_exp_r[3] * c.y + m_mat_exp_r[4] * c.z;
    c.z = m_mat_exp_r[5] * c.z;

    global_box.setL(make_scalar3(a.x, b.y, c.z));
    Scalar xy = b.x / b.y;

    Scalar xz(0.0);
    Scalar yz(0.0);

    if (!is_two_dimensions)
        {
        xz = c.x / c.z;
        yz = c.y / c.z;
        }

    global_box.setTiltFactors(xy, xz, yz);

    // set global box
    m_pdata->setGlobalBox(global_box);
    m_V = global_box.getVolume(is_two_dimensions); // volume

    if (m_rescale_all)
        {
        // rescale all particle positions
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);

        unsigned int nparticles = m_pdata->getN();

        for (unsigned int i = 0; i < nparticles; i++)
            {
            Scalar3 r = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);

            r.x = m_mat_exp_r[0] * r.x + m_mat_exp_r[1] * r.y + m_mat_exp_r[2] * r.z;
            r.y = m_mat_exp_r[3] * r.y + m_mat_exp_r[4] * r.z;
            r.z = m_mat_exp_r[5] * r.z;

            h_pos.data[i].x = r.x;
            h_pos.data[i].y = r.y;
            h_pos.data[i].z = r.z;
            }
        }

        {
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                     access_location::host,
                                     access_mode::read);
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);

        // precompute loop invariant quantity
        Scalar xi_trans = v.variable[1];
        Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (xi_trans + mtk) * m_deltaT);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
            Scalar3 accel = h_accel.data[j];
            Scalar3 r = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

            // advance velocity
            v += m_deltaT / Scalar(2.0) * accel;

            // apply barostat by multiplying with matrix exponential
            v.x = m_mat_exp_v[0] * v.x + m_mat_exp_v[1] * v.y + m_mat_exp_v[2] * v.z;
            v.y = m_mat_exp_v[3] * v.y + m_mat_exp_v[4] * v.z;
            v.z = m_mat_exp_v[5] * v.z;

            // apply thermostat update of velocity
            v *= exp_thermo_fac;

            if (!m_rescale_all)
                {
                r.x = m_mat_exp_r[0] * r.x + m_mat_exp_r[1] * r.y + m_mat_exp_r[2] * r.z;
                r.y = m_mat_exp_r[3] * r.y + m_mat_exp_r[4] * r.z;
                r.z = m_mat_exp_r[5] * r.z;
                }

            r.x += m_mat_exp_r_int[0] * v.x + m_mat_exp_r_int[1] * v.y + m_mat_exp_r_int[2] * v.z;
            r.y += m_mat_exp_r_int[3] * v.y + m_mat_exp_r_int[4] * v.z;
            r.z += m_mat_exp_r_int[5] * v.z;

            // store velocity
            h_vel.data[j].x = v.x;
            h_vel.data[j].y = v.y;
            h_vel.data[j].z = v.z;

            // store position
            h_pos.data[j].x = r.x;
            h_pos.data[j].y = r.y;
            h_pos.data[j].z = r.z;
            }
        } // end of GPUArray scope

    // Get new local box
    BoxDim box = m_pdata->getBox();

        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);

        // Wrap particles
        for (unsigned int j = 0; j < m_pdata->getN(); j++)
            box.wrap(h_pos.data[j], h_image.data[j]);
        }

    // Integration of angular degrees of freedom using symplectic and
    // time-reversal symmetric integration scheme of Miller et al., extended by thermostat
    if (m_aniso)
        {
        // precompute loop invariant quantity
        Scalar xi_rot = v.variable[8];
        Scalar exp_thermo_fac_rot = exp(-(xi_rot + mtk) * m_deltaT / Scalar(2.0));

        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::host,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::host,
                                          access_mode::read);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::host,
                                       access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            quat<Scalar> q(h_orientation.data[j]);
            quat<Scalar> p(h_angmom.data[j]);
            vec3<Scalar> t(h_net_torque.data[j]);
            vec3<Scalar> I(h_inertia.data[j]);

            // rotate torque into principal frame
            t = rotate(conj(q), t);

            // check for zero moment of inertia
            bool x_zero, y_zero, z_zero;
            x_zero = (I.x < EPSILON);
            y_zero = (I.y < EPSILON);
            z_zero = (I.z < EPSILON);

            // ignore torque component along an axis for which the moment of inertia zero
            if (x_zero)
                t.x = 0;
            if (y_zero)
                t.y = 0;
            if (z_zero)
                t.z = 0;

            // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
            p += m_deltaT * q * t;

            // apply thermostat
            p = p * exp_thermo_fac_rot;

            quat<Scalar> p1, p2, p3; // permutated quaternions
            quat<Scalar> q1, q2, q3;
            Scalar phi1, cphi1, sphi1;
            Scalar phi2, cphi2, sphi2;
            Scalar phi3, cphi3, sphi3;

            if (!z_zero)
                {
                p3 = quat<Scalar>(-p.v.z, vec3<Scalar>(p.v.y, -p.v.x, p.s));
                q3 = quat<Scalar>(-q.v.z, vec3<Scalar>(q.v.y, -q.v.x, q.s));
                phi3 = Scalar(1. / 4.) / I.z * dot(p, q3);
                cphi3 = slow::cos(Scalar(1. / 2.) * m_deltaT * phi3);
                sphi3 = slow::sin(Scalar(1. / 2.) * m_deltaT * phi3);

                p = cphi3 * p + sphi3 * p3;
                q = cphi3 * q + sphi3 * q3;
                }

            if (!y_zero)
                {
                p2 = quat<Scalar>(-p.v.y, vec3<Scalar>(-p.v.z, p.s, p.v.x));
                q2 = quat<Scalar>(-q.v.y, vec3<Scalar>(-q.v.z, q.s, q.v.x));
                phi2 = Scalar(1. / 4.) / I.y * dot(p, q2);
                cphi2 = slow::cos(Scalar(1. / 2.) * m_deltaT * phi2);
                sphi2 = slow::sin(Scalar(1. / 2.) * m_deltaT * phi2);

                p = cphi2 * p + sphi2 * p2;
                q = cphi2 * q + sphi2 * q2;
                }

            if (!x_zero)
                {
                p1 = quat<Scalar>(-p.v.x, vec3<Scalar>(p.s, p.v.z, -p.v.y));
                q1 = quat<Scalar>(-q.v.x, vec3<Scalar>(q.s, q.v.z, -q.v.y));
                phi1 = Scalar(1. / 4.) / I.x * dot(p, q1);
                cphi1 = slow::cos(m_deltaT * phi1);
                sphi1 = slow::sin(m_deltaT * phi1);

                p = cphi1 * p + sphi1 * p1;
                q = cphi1 * q + sphi1 * q1;
                }

            if (!y_zero)
                {
                p2 = quat<Scalar>(-p.v.y, vec3<Scalar>(-p.v.z, p.s, p.v.x));
                q2 = quat<Scalar>(-q.v.y, vec3<Scalar>(-q.v.z, q.s, q.v.x));
                phi2 = Scalar(1. / 4.) / I.y * dot(p, q2);
                cphi2 = slow::cos(Scalar(1. / 2.) * m_deltaT * phi2);
                sphi2 = slow::sin(Scalar(1. / 2.) * m_deltaT * phi2);

                p = cphi2 * p + sphi2 * p2;
                q = cphi2 * q + sphi2 * q2;
                }

            if (!z_zero)
                {
                p3 = quat<Scalar>(-p.v.z, vec3<Scalar>(p.v.y, -p.v.x, p.s));
                q3 = quat<Scalar>(-q.v.z, vec3<Scalar>(q.v.y, -q.v.x, q.s));
                phi3 = Scalar(1. / 4.) / I.z * dot(p, q3);
                cphi3 = slow::cos(Scalar(1. / 2.) * m_deltaT * phi3);
                sphi3 = slow::sin(Scalar(1. / 2.) * m_deltaT * phi3);

                p = cphi3 * p + sphi3 * p3;
                q = cphi3 * q + sphi3 * q3;
                }

            // renormalize (improves stability)
            q = q * (Scalar(1.0) / slow::sqrt(norm2(q)));

            h_orientation.data[j] = quat_to_scalar4(q);
            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    if (!m_nph)
        {
        // propagate thermostat variables forward
        advanceThermostat(timestep);
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // broadcast integrator variables from rank 0 to other processors
        v = getIntegratorVariables();
        MPI_Bcast(&v.variable.front(), 10, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        setIntegratorVariables(v);
        }
#endif

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNPTMTK::integrateStepTwo(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("NPT step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar nuxx = v.variable[2]; // Barostat tensor, xx component
    Scalar nuyy = v.variable[5]; // Barostat tensor, yy component
    Scalar nuzz = v.variable[7]; // Barostat tensor, zz component

        {
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                     access_location::host,
                                     access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

        // precompute loop invariant quantity
        Scalar xi_trans = v.variable[1];
        Scalar mtk = (nuxx + nuyy + nuzz) / (Scalar)m_ndof;
        Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (xi_trans + mtk) * m_deltaT);

        // perform second half step of NPT integration
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            // first, calculate acceleration from the net force
            Scalar m = h_vel.data[j].w;
            Scalar minv = Scalar(1.0) / m;
            h_accel.data[j].x = h_net_force.data[j].x * minv;
            h_accel.data[j].y = h_net_force.data[j].y * minv;
            h_accel.data[j].z = h_net_force.data[j].z * minv;

            Scalar3 accel = make_scalar3(h_accel.data[j].x, h_accel.data[j].y, h_accel.data[j].z);

            // update velocity by multiplication with upper triangular matrix
            Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);

            // apply thermostat
            v = v * exp_thermo_fac;

            // apply barostat by multiplying with matrix exponential
            v.x = m_mat_exp_v[0] * v.x + m_mat_exp_v[1] * v.y + m_mat_exp_v[2] * v.z;
            v.y = m_mat_exp_v[3] * v.y + m_mat_exp_v[4] * v.z;
            v.z = m_mat_exp_v[5] * v.z;

            // advance velocity
            v += m_deltaT / Scalar(2.0) * accel;

            // store velocity
            h_vel.data[j].x = v.x;
            h_vel.data[j].y = v.y;
            h_vel.data[j].z = v.z;
            }

        if (m_aniso)
            {
            // angular degrees of freedom
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                               access_location::host,
                                               access_mode::read);
            ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                          access_location::host,
                                          access_mode::readwrite);
            ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),
                                              access_location::host,
                                              access_mode::read);
            ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                           access_location::host,
                                           access_mode::read);

            // precompute loop invariant quantity
            Scalar xi_rot = v.variable[8];
            Scalar exp_thermo_fac_rot = exp(-(xi_rot + mtk) * m_deltaT / Scalar(2.0));

            // apply rotational (NO_SQUISH) equations of motion
            for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
                {
                unsigned int j = m_group->getMemberIndex(group_idx);

                quat<Scalar> q(h_orientation.data[j]);
                quat<Scalar> p(h_angmom.data[j]);
                vec3<Scalar> t(h_net_torque.data[j]);
                vec3<Scalar> I(h_inertia.data[j]);

                // rotate torque into principal frame
                t = rotate(conj(q), t);

                // check for zero moment of inertia
                bool x_zero, y_zero, z_zero;
                x_zero = (I.x < EPSILON);
                y_zero = (I.y < EPSILON);
                z_zero = (I.z < EPSILON);

                // ignore torque component along an axis for which the moment of inertia zero
                if (x_zero)
                    t.x = 0;
                if (y_zero)
                    t.y = 0;
                if (z_zero)
                    t.z = 0;

                // thermostat angular degrees of freedom
                p = p * exp_thermo_fac_rot;

                // advance p(t+deltaT/2)->p(t+deltaT)
                p += m_deltaT * q * t;

                h_angmom.data[j] = quat_to_scalar4(p);
                }
            }
        } // end GPUArray scope

    // advance barostat (nuxx, nuyy, nuzz) half a time step
    advanceBarostat(timestep + 1);

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param nuxx Barostat matrix, xx element
    \param nuxy xy element
    \param nuxz xz element
    \param nuyy yy element
    \param nuyz yz element
    \param nuzz zz element
*/
void TwoStepNPTMTK::updatePropagator(Scalar nuxx,
                                     Scalar nuxy,
                                     Scalar nuxz,
                                     Scalar nuyy,
                                     Scalar nuyz,
                                     Scalar nuzz)
    {
    // calculate some factors needed for the update matrix
    Scalar3 v_fac = make_scalar3(-Scalar(1.0 / 4.0) * nuxx,
                                 -Scalar(1.0 / 4.0) * nuyy,
                                 -Scalar(1.0 / 4.0) * nuzz);
    Scalar3 exp_v_fac_2 = make_scalar3(exp(Scalar(2.0) * v_fac.x * m_deltaT),
                                       exp(Scalar(2.0) * v_fac.y * m_deltaT),
                                       exp(Scalar(2.0) * v_fac.z * m_deltaT));

    Scalar3 r_fac = make_scalar3(Scalar(1.0 / 2.0) * nuxx,
                                 Scalar(1.0 / 2.0) * nuyy,
                                 Scalar(1.0 / 2.0) * nuzz);
    Scalar3 exp_r_fac = make_scalar3(exp(Scalar(1.0 / 2.0) * nuxx * m_deltaT),
                                     exp(Scalar(1.0 / 2.0) * nuyy * m_deltaT),
                                     exp(Scalar(1.0 / 2.0) * nuzz * m_deltaT));
    Scalar3 exp_r_fac_2
        = make_scalar3(exp(nuxx * m_deltaT), exp(nuyy * m_deltaT), exp(nuzz * m_deltaT));

    // Calculate power series approximations of analytical functions entering the update equations

    Scalar3 arg_v = v_fac * m_deltaT;
    Scalar3 arg_r = r_fac * m_deltaT;

    // Calculate function f = sinh(x)/x
    Scalar3 f_v = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 f_r = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 term_v = make_scalar3(1.0, 1.0, 1.0);
    Scalar3 term_r = make_scalar3(1.0, 1.0, 1.0);

    for (unsigned int i = 0; i < 6; i++)
        {
        f_v += f_coeff[i] * term_v;
        f_r += f_coeff[i] * term_r;
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate function g = cth(x) - 1/x
    Scalar3 g_v = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 g_r = make_scalar3(0.0, 0.0, 0.0);

    term_v = arg_v;
    term_r = arg_r;

    for (unsigned int i = 0; i < 5; i++)
        {
        g_v += g_coeff[i] * term_v;
        g_r += g_coeff[i] * term_r;
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate function h = -1/sinh^2(x) + 1/x^2
    Scalar3 h_v = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 h_r = make_scalar3(0.0, 0.0, 0.0);

    term_v = term_r = make_scalar3(1.0, 1.0, 1.0);

    for (unsigned int i = 0; i < 6; i++)
        {
        h_v += h_coeff[i] * term_v;
        h_r += h_coeff[i] * term_r;

        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate matrix exponentials for upper triangular barostat matrix
    /* These are approximations accurate up to and including delta_t^2.
       They are fully time reversible  */

    // Matrix exp. for velocity update
    m_mat_exp_v[0] = exp_v_fac_2.x;                                                          // xx
    m_mat_exp_v[1] = -m_deltaT * Scalar(1.0 / 4.0) * nuxy * (exp_v_fac_2.x + exp_v_fac_2.y); // xy
    m_mat_exp_v[2] = -m_deltaT * Scalar(1.0 / 4.0) * nuxz * (exp_v_fac_2.x + exp_v_fac_2.z)
                     + m_deltaT * m_deltaT * Scalar(1.0 / 32.0) * nuxy * nuyz
                           * (exp_v_fac_2.x + Scalar(2.0) * exp_v_fac_2.y + exp_v_fac_2.z);  // xz
    m_mat_exp_v[3] = exp_v_fac_2.y;                                                          // yy
    m_mat_exp_v[4] = -m_deltaT * Scalar(1.0 / 4.0) * nuyz * (exp_v_fac_2.y + exp_v_fac_2.z); // yz
    m_mat_exp_v[5] = exp_v_fac_2.z;                                                          // zz

    // Matrix exp. for position update
    m_mat_exp_r[0] = exp_r_fac_2.x;                                                         // xx
    m_mat_exp_r[1] = m_deltaT * Scalar(1.0 / 2.0) * nuxy * (exp_r_fac_2.x + exp_r_fac_2.y); // xy
    m_mat_exp_r[2] = m_deltaT * Scalar(1.0 / 2.0) * nuxz * (exp_r_fac_2.x + exp_r_fac_2.z)
                     + m_deltaT * m_deltaT * Scalar(1.0 / 8.0) * nuxy * nuyz
                           * (exp_r_fac_2.x + Scalar(2.0) * exp_r_fac_2.y + exp_r_fac_2.z); // xz
    m_mat_exp_r[3] = exp_r_fac_2.y;                                                         // yy
    m_mat_exp_r[4] = m_deltaT * Scalar(1.0 / 2.0) * nuyz * (exp_r_fac_2.y + exp_r_fac_2.z); // yz
    m_mat_exp_r[5] = exp_r_fac_2.z;                                                         // zz

    // integrated matrix exp. for position update
    Scalar3 xz_fac_r = make_scalar3((Scalar(1.0) + g_r.x) * (Scalar(1.0) + g_r.x) + h_r.x,
                                    (Scalar(1.0) + g_r.y) * (Scalar(1.0) + g_r.y) + h_r.y,
                                    (Scalar(1.0) + g_r.z) * (Scalar(1.0) + g_r.z) + h_r.z);

    m_mat_exp_r_int[0] = m_deltaT * exp_r_fac.x * f_r.x; // xx
    m_mat_exp_r_int[1] = m_deltaT * m_deltaT * nuxy * Scalar(1.0 / 4.0)
                         * (exp_r_fac.x * f_r.x * (Scalar(1.0) + g_r.x)
                            + exp_r_fac.y * f_r.y * (Scalar(1.0) + g_r.y)); // xy
    m_mat_exp_r_int[2]
        = m_deltaT * m_deltaT * nuxz * Scalar(1.0 / 4.0)
              * (exp_r_fac.x * f_r.x * (Scalar(1.0) + g_r.x)
                 + exp_r_fac.z * f_r.z * (Scalar(1.0) + g_r.z))
          + m_deltaT * m_deltaT * m_deltaT * nuxy * nuyz * Scalar(1.0 / 32.0)
                * (exp_r_fac.x * f_r.x * xz_fac_r.x + Scalar(2.0) * exp_r_fac.y * f_r.y * xz_fac_r.y
                   + exp_r_fac.z * f_r.z * xz_fac_r.z);  // xz
    m_mat_exp_r_int[3] = m_deltaT * exp_r_fac.y * f_r.y; // yy
    m_mat_exp_r_int[4] = m_deltaT * m_deltaT * nuyz * Scalar(1.0 / 4.0)
                         * (exp_r_fac.y * f_r.y * (Scalar(1.0) + g_r.y)
                            + exp_r_fac.z * f_r.z * (Scalar(1.0) + g_r.z)); // yz
    m_mat_exp_r_int[5] = m_deltaT * exp_r_fac.z * f_r.z;                    // zz
    }

// Set Flags from 6 element boolean tuple named box_df to integer flag
void TwoStepNPTMTK::setFlags(const std::vector<bool>& value)
    {
    bool is_three_dimensions = m_sysdef->getNDimensions() == 3;
    int flags = 0;
    if (value[0])
        flags |= int(baroFlags::baro_x);
    if (value[1])
        flags |= int(baroFlags::baro_y);
    if (value[2] && is_three_dimensions)
        flags |= int(baroFlags::baro_z);
    if (value[3])
        flags |= int(baroFlags::baro_xy);
    if (value[4] && is_three_dimensions)
        flags |= int(baroFlags::baro_xz);
    if (value[5] && is_three_dimensions)
        flags |= int(baroFlags::baro_yz);
    m_flags = flags;
    }

// Get Flags from integer flag to 6 element boolean tuple
std::vector<bool> TwoStepNPTMTK::getFlags()
    {
    std::vector<bool> result;
    result.push_back(m_flags & baro_x);
    result.push_back(m_flags & baro_y);
    result.push_back(m_flags & baro_z);
    result.push_back(m_flags & baro_xy);
    result.push_back(m_flags & baro_xz);
    result.push_back(m_flags & baro_yz);
    return result;
    }

//! Helper function to advance the barostat parameters
void TwoStepNPTMTK::advanceBarostat(uint64_t timestep)
    {
    // compute thermodynamic properties at full time step
    m_thermo_full_step->compute(timestep);

    // compute pressure for the next half time step
    PressureTensor P = m_thermo_full_step->getPressureTensor();

    if (std::isnan(P.xx) || std::isnan(P.xy) || std::isnan(P.xz) || std::isnan(P.yy)
        || std::isnan(P.yz) || std::isnan(P.zz))
        {
        P.xx = (*m_S[0])(timestep);
        P.yy = (*m_S[1])(timestep);
        P.zz = (*m_S[2])(timestep);
        P.yz = (*m_S[3])(timestep);
        P.xz = (*m_S[4])(timestep);
        P.xy = (*m_S[5])(timestep);
        }

    // advance barostat (nuxx, nuyy, nuzz) half a time step
    // Martyna-Tobias-Klein correction
    unsigned int d = m_sysdef->getNDimensions();
    Scalar W = (Scalar)(m_ndof + d) / (Scalar)d * (*m_T)(timestep)*m_tauS * m_tauS;
    Scalar mtk_term = Scalar(2.0) * m_thermo_full_step->getTranslationalKineticEnergy();
    mtk_term *= Scalar(1.0 / 2.0) * m_deltaT / (Scalar)m_ndof / W;

    couplingMode couple = getRelevantCouplings();

    // couple diagonal elements of pressure tensor together
    Scalar3 P_diag = make_scalar3(0.0, 0.0, 0.0);

    if (couple == couple_none)
        {
        P_diag.x = P.xx;
        P_diag.y = P.yy;
        P_diag.z = P.zz;
        }
    else if (couple == couple_xy)
        {
        P_diag.x = Scalar(1.0 / 2.0) * (P.xx + P.yy);
        P_diag.y = Scalar(1.0 / 2.0) * (P.xx + P.yy);
        P_diag.z = P.zz;
        }
    else if (couple == couple_xz)
        {
        P_diag.x = Scalar(1.0 / 2.0) * (P.xx + P.zz);
        P_diag.y = P.yy;
        P_diag.z = Scalar(1.0 / 2.0) * (P.xx + P.zz);
        }
    else if (couple == couple_yz)
        {
        P_diag.x = P.xx;
        P_diag.y = Scalar(1.0 / 2.0) * (P.yy + P.zz);
        P_diag.z = Scalar(1.0 / 2.0) * (P.yy + P.zz);
        }
    else if (couple == couple_xyz)
        {
        Scalar P_iso = Scalar(1.0 / 3.0) * (P.xx + P.yy + P.zz);
        P_diag.x = P_diag.y = P_diag.z = P_iso;
        }
    else
        {
        m_exec_conf->msg->error() << "integrate.npt: Invalid coupling mode." << std::endl
                                  << std::endl;
        throw std::runtime_error("Error in NPT integration");
        }

    // update barostat matrix
    IntegratorVariables v = getIntegratorVariables();
    Scalar& nuxx = v.variable[2]; // Barostat tensor, xx component
    Scalar& nuxy = v.variable[3]; // Barostat tensor, xy component
    Scalar& nuxz = v.variable[4]; // Barostat tensor, xz component
    Scalar& nuyy = v.variable[5]; // Barostat tensor, yy component
    Scalar& nuyz = v.variable[6]; // Barostat tensor, yz component
    Scalar& nuzz = v.variable[7]; // Barostat tensor, zz component

    if (m_flags & baro_x)
        {
        nuxx
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.x - (*m_S[0])(timestep)) + mtk_term;
        nuxx -= m_gamma * nuxx;
        }

    if (m_flags & baro_xy)
        {
        nuxy += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.xy - (*m_S[5])(timestep));
        nuxy -= m_gamma * nuxy;
        }

    if (m_flags & baro_xz)
        {
        nuxz += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.xz - (*m_S[4])(timestep));
        nuxz -= m_gamma * nuxz;
        }

    if (m_flags & baro_y)
        {
        nuyy
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.y - (*m_S[1])(timestep)) + mtk_term;
        nuyy -= m_gamma * nuyy;
        }

    if (m_flags & baro_yz)
        {
        nuyz += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.yz - (*m_S[3])(timestep));
        nuyz -= m_gamma * nuyz;
        }

    if (m_flags & baro_z)
        {
        nuzz
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.z - (*m_S[2])(timestep)) + mtk_term;
        nuzz -= m_gamma * nuzz;
        }

    // store integrator variables
    setIntegratorVariables(v);
    }

void TwoStepNPTMTK::advanceThermostat(uint64_t timestep)
    {
    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];
    Scalar& xi = v.variable[1];

    // compute the current thermodynamic properties
    m_thermo_half_step->compute(timestep);

    Scalar curr_T_trans = m_thermo_half_step->getTranslationalTemperature();
    Scalar T = (*m_T)(timestep);

    // update the state variables Xi and eta
    Scalar xi_prime
        = xi + Scalar(1.0 / 2.0) * m_deltaT / m_tau / m_tau * (curr_T_trans / T - Scalar(1.0));
    xi = xi_prime + Scalar(1.0 / 2.0) * m_deltaT / m_tau / m_tau * (curr_T_trans / T - Scalar(1.0));
    eta += xi_prime * m_deltaT;

    if (m_aniso)
        {
        // update thermostat for rotational DOF
        Scalar& xi_rot = v.variable[8];
        Scalar& eta_rot = v.variable[9];

        Scalar curr_ke_rot = m_thermo_half_step->getRotationalKineticEnergy();
        Scalar ndof_rot = m_group->getRotationalDOF();

        Scalar xi_prime_rot = xi_rot
                              + Scalar(1.0 / 2.0) * m_deltaT / m_tau / m_tau
                                    * (Scalar(2.0) * curr_ke_rot / ndof_rot / T - Scalar(1.0));
        xi_rot = xi_prime_rot
                 + Scalar(1.0 / 2.0) * m_deltaT / m_tau / m_tau
                       * (Scalar(2.0) * curr_ke_rot / ndof_rot / T - Scalar(1.0));

        eta_rot += xi_prime_rot * m_deltaT;
        }

    setIntegratorVariables(v);
    }

void TwoStepNPTMTK::setCouple(const std::string& value)
    {
    bool is_two_dimensions = m_sysdef->getNDimensions() == 2;

    if (is_two_dimensions)
        {
        if (value == "none")
            {
            m_couple = couple_none;
            }
        else if (value == "xy")
            {
            m_couple = couple_xy;
            }
        else
            {
            throw std::invalid_argument("Invalid coupling mode " + value + " for 2D simulations.");
            }
        }
    else
        {
        if (value == "none")
            {
            m_couple = couple_none;
            }
        else if (value == "xy")
            {
            m_couple = couple_xy;
            }
        else if (value == "xz")
            {
            m_couple = couple_xz;
            }
        else if (value == "yz")
            {
            m_couple = couple_yz;
            }
        else if (value == "xyz")
            {
            m_couple = couple_xyz;
            }
        else
            {
            throw std::invalid_argument("Invalid coupling mode " + value);
            }
        }
    }

std::string TwoStepNPTMTK::getCouple()
    {
    std::string couple;

    switch (m_couple)
        {
    case couple_none:
        couple = "none";
        break;
    case couple_xy:
        couple = "xy";
        break;
    case couple_xz:
        couple = "xz";
        break;
    case couple_yz:
        couple = "yz";
        break;
    case couple_xyz:
        couple = "xyz";
        }
    return couple;
    }

TwoStepNPTMTK::couplingMode TwoStepNPTMTK::getRelevantCouplings()
    {
    couplingMode couple = m_couple;
    // disable irrelevant couplings
    if (!(m_flags & baro_x))
        {
        if (couple == couple_xyz)
            {
            couple = couple_yz;
            }
        if (couple == couple_xy || couple == couple_xz)
            {
            couple = couple_none;
            }
        }
    if (!(m_flags & baro_y))
        {
        if (couple == couple_xyz)
            {
            couple = couple_xz;
            }
        if (couple == couple_yz || couple == couple_xy)
            {
            couple = couple_none;
            }
        }
    if (!(m_flags & baro_z))
        {
        if (couple == couple_xyz)
            {
            couple = couple_xy;
            }
        if (couple == couple_yz || couple == couple_xz)
            {
            couple = couple_none;
            }
        }
    return couple;
    }

void TwoStepNPTMTK::thermalizeThermostatAndBarostatDOF(uint64_t timestep)
    {
    m_exec_conf->msg->notice(6) << "TwoStepNPTMTK randomizing thermostat and barostat DOF"
                                << std::endl;

    IntegratorVariables v = getIntegratorVariables();

    unsigned int instance_id = 0;
    if (m_group->getNumMembersGlobal() > 0)
        instance_id = m_group->getMemberTag(0);

    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::TwoStepNPTMTK, timestep, m_sysdef->getSeed()),
        hoomd::Counter(instance_id));

    bool master = m_exec_conf->getRank() == 0;

    if (!m_nph)
        {
        // randomize thermostat variables
        Scalar& xi = v.variable[1];

        Scalar g = m_group->getTranslationalDOF();
        Scalar sigmasq_t = Scalar(1.0) / (g * m_tau * m_tau);

        if (master)
            {
            // draw a random Gaussian thermostat variable on rank 0
            xi = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_t))(rng);
            }

        if (m_aniso)
            {
            // update thermostat for rotational DOF
            Scalar& xi_rot = v.variable[8];
            Scalar sigmasq_r = Scalar(1.0) / ((Scalar)m_group->getRotationalDOF() * m_tau * m_tau);

            if (master)
                {
                xi_rot = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_r))(rng);
                }
            }
        }

    // randomize barostat variables
    Scalar& nuxx = v.variable[2]; // Barostat tensor, xx component
    Scalar& nuxy = v.variable[3]; // Barostat tensor, xy component
    Scalar& nuxz = v.variable[4]; // Barostat tensor, xz component
    Scalar& nuyy = v.variable[5]; // Barostat tensor, yy component
    Scalar& nuyz = v.variable[6]; // Barostat tensor, yz component
    Scalar& nuzz = v.variable[7]; // Barostat tensor, zz component

    unsigned int d = m_sysdef->getNDimensions();
    Scalar sigmasq_baro = Scalar(1.0) / ((Scalar)(m_ndof + d) / (Scalar)d * m_tauS * m_tauS);

    if (master)
        {
        if (m_flags & baro_x)
            {
            nuxx = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_xy)
            {
            nuxy = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_xz)
            {
            nuxz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_y)
            {
            nuyy = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_yz)
            {
            nuyz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_z)
            {
            nuzz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        // couple box degrees of freedom
        couplingMode couple = getRelevantCouplings();

        switch (couple)
            {
        case couple_none:
            break;
        case couple_xy:
            nuyy = nuxx;
            break;
        case couple_xz:
            nuzz = nuxx;
            break;
        case couple_yz:
            nuyy = nuzz;
            break;
        case couple_xyz:
            nuxx = nuyy = nuzz;
            break;
        default:
            m_exec_conf->msg->error() << "integrate.npt: Invalid coupling mode." << std::endl
                                      << std::endl;
            throw std::runtime_error("Error in NPT integration");
            }
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&v.variable.front(), 10, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    setIntegratorVariables(v);
    }

pybind11::tuple TwoStepNPTMTK::getTranslationalThermostatDOF()
    {
    pybind11::list result;
    IntegratorVariables v = getIntegratorVariables();

    Scalar& eta = v.variable[0];
    Scalar& xi = v.variable[1];

    result.append(xi);
    result.append(eta);

    return pybind11::tuple(result);
    }

void TwoStepNPTMTK::setTranslationalThermostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 2)
        {
        throw std::length_error("translational_thermostat_dof must have length 2");
        }

    IntegratorVariables vars = getIntegratorVariables();

    Scalar& eta = vars.variable[0];
    Scalar& xi = vars.variable[1];

    xi = pybind11::cast<Scalar>(v[0]);
    eta = pybind11::cast<Scalar>(v[1]);

    setIntegratorVariables(vars);
    }

pybind11::tuple TwoStepNPTMTK::getRotationalThermostatDOF()
    {
    pybind11::list result;
    IntegratorVariables v = getIntegratorVariables();

    Scalar& xi_rot = v.variable[8];
    Scalar& eta_rot = v.variable[9];

    result.append(xi_rot);
    result.append(eta_rot);

    return pybind11::tuple(result);
    }

void TwoStepNPTMTK::setRotationalThermostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 2)
        {
        throw std::length_error("rotational_thermostat_dof must have length 2");
        }

    IntegratorVariables vars = getIntegratorVariables();

    Scalar& xi_rot = vars.variable[8];
    Scalar& eta_rot = vars.variable[9];

    xi_rot = pybind11::cast<Scalar>(v[0]);
    eta_rot = pybind11::cast<Scalar>(v[1]);

    setIntegratorVariables(vars);
    }

Scalar TwoStepNPTMTK::getThermostatEnergy(uint64_t timestep)
    {
    IntegratorVariables integrator_variables = getIntegratorVariables();
    Scalar eta = integrator_variables.variable[0];
    Scalar xi = integrator_variables.variable[1];

    Scalar thermostat_energy = m_group->getTranslationalDOF() * (*m_T)(timestep)
                               * (eta + m_tau * m_tau * xi * xi / Scalar(2.0));

    if (m_aniso)
        {
        Scalar xi_rot = integrator_variables.variable[8];
        Scalar eta_rot = integrator_variables.variable[9];
        thermostat_energy += m_group->getRotationalDOF() * (*m_T)(timestep)
                             * (eta_rot + m_tau * m_tau * xi_rot * xi_rot / Scalar(2.0));
        }

    return thermostat_energy;
    }

pybind11::tuple TwoStepNPTMTK::getBarostatDOF()
    {
    pybind11::list result;
    IntegratorVariables v = getIntegratorVariables();

    for (size_t i = 0; i < 6; i++)
        {
        result.append(v.variable[i + 2]);
        }

    return pybind11::tuple(result);
    }

void TwoStepNPTMTK::setBarostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 6)
        {
        throw std::length_error("barostat_dof must have length 6");
        }

    IntegratorVariables vars = getIntegratorVariables();

    for (size_t i = 0; i < 6; i++)
        {
        vars.variable[i + 2] = pybind11::cast<Scalar>(v[i]);
        }

    setIntegratorVariables(vars);
    }

Scalar TwoStepNPTMTK::getBarostatEnergy(uint64_t timestep)
    {
    IntegratorVariables integrator_variables = getIntegratorVariables();

    Scalar nu_xx = integrator_variables.variable[2]; // Barostat tensor, xx component
    Scalar nu_xy = integrator_variables.variable[3]; // Barostat tensor, xy component
    Scalar nu_xz = integrator_variables.variable[4]; // Barostat tensor, xz component
    Scalar nu_yy = integrator_variables.variable[5]; // Barostat tensor, yy component
    Scalar nu_yz = integrator_variables.variable[6]; // Barostat tensor, yz component
    Scalar nu_zz = integrator_variables.variable[7]; // Barostat tensor, zz component

    unsigned int d = m_sysdef->getNDimensions();
    Scalar W = static_cast<Scalar>(m_ndof + d) / static_cast<Scalar>(d) * (*m_T)(timestep)*m_tauS
               * m_tauS;

    Scalar barostat_energy = W
                             * (nu_xx * nu_xx + nu_yy * nu_yy + nu_zz * nu_zz   // Normal
                                + nu_xy * nu_xy + nu_xz * nu_xz + nu_yz * nu_yz // Shear
                                )
                             / Scalar(2.0);

    return barostat_energy;
    }

namespace detail
    {
void export_TwoStepNPTMTK(pybind11::module& m)
    {
    pybind11::class_<TwoStepNPTMTK, IntegrationMethodTwoStep, std::shared_ptr<TwoStepNPTMTK>>
        twostepnptmtk(m, "TwoStepNPTMTK");
    twostepnptmtk
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            Scalar,
                            std::shared_ptr<Variant>,
                            const std::vector<std::shared_ptr<Variant>>&,
                            const string&,
                            const std::vector<bool>&,
                            const bool>())
        .def_property("kT", &TwoStepNPTMTK::getT, &TwoStepNPTMTK::setT)
        .def_property("S", &TwoStepNPTMTK::getS, &TwoStepNPTMTK::setS)
        .def_property("tau", &TwoStepNPTMTK::getTau, &TwoStepNPTMTK::setTau)
        .def_property("tauS", &TwoStepNPTMTK::getTauS, &TwoStepNPTMTK::setTauS)
        .def_property("couple", &TwoStepNPTMTK::getCouple, &TwoStepNPTMTK::setCouple)
        .def_property("box_dof", &TwoStepNPTMTK::getFlags, &TwoStepNPTMTK::setFlags)
        .def_property("rescale_all", &TwoStepNPTMTK::getRescaleAll, &TwoStepNPTMTK::setRescaleAll)
        .def_property("gamma", &TwoStepNPTMTK::getGamma, &TwoStepNPTMTK::setGamma)
        .def("thermalizeThermostatAndBarostatDOF",
             &TwoStepNPTMTK::thermalizeThermostatAndBarostatDOF)
        .def_property("translational_thermostat_dof",
                      &TwoStepNPTMTK::getTranslationalThermostatDOF,
                      &TwoStepNPTMTK::setTranslationalThermostatDOF)
        .def_property("rotational_thermostat_dof",
                      &TwoStepNPTMTK::getRotationalThermostatDOF,
                      &TwoStepNPTMTK::setRotationalThermostatDOF)
        .def_property("barostat_dof",
                      &TwoStepNPTMTK::getBarostatDOF,
                      &TwoStepNPTMTK::setBarostatDOF)
        .def("getThermostatEnergy", &TwoStepNPTMTK::getThermostatEnergy)
        .def("getBarostatEnergy", &TwoStepNPTMTK::getBarostatEnergy);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
