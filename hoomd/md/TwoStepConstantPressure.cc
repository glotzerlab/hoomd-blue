// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepConstantPressure.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

namespace hoomd::md
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

TwoStepConstantPressure::TwoStepConstantPressure(std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<ParticleGroup> group,
                                                 std::shared_ptr<ComputeThermo> thermo_full_step,
                                                 Scalar tauS,
                                                 const std::vector<std::shared_ptr<Variant>>& S,
                                                 const std::string& couple,
                                                 const std::vector<bool>& flags,
                                                 std::shared_ptr<Thermostat> thermostat,
                                                 Scalar gamma)
    : IntegrationMethodTwoStep(sysdef, group), m_S(S), m_tauS(tauS), m_ndof(0), m_gamma(gamma),
      m_thermostat(thermostat), m_thermo_full_step(thermo_full_step), m_rescale_all(false)
    {
    setCouple(couple);
    setFlags(flags);

    if (m_flags == 0)
        {
        m_exec_conf->msg->warning() << "ConstantPressure: No box degrees of freedom." << std::endl;
        }

    bool is_two_dimensions = m_sysdef->getNDimensions() == 2;
    m_V = m_pdata->getGlobalBox().getVolume(is_two_dimensions); // volume
    }

void TwoStepConstantPressure::setCouple(const std::string& value)
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

std::string TwoStepConstantPressure::getCouple()
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

TwoStepConstantPressure::couplingMode TwoStepConstantPressure::getRelevantCouplings()
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

// Set Flags from 6 element boolean tuple named box_df to integer flag
void TwoStepConstantPressure::setFlags(const std::vector<bool>& value)
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
std::vector<bool> TwoStepConstantPressure::getFlags()
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

void TwoStepConstantPressure::integrateStepOne(uint64_t timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        throw std::runtime_error("Invalid NPT coupling mode.");
        }

    // update box dimensions
    bool is_two_dimensions = m_sysdef->getNDimensions() == 2;

    m_V = m_pdata->getGlobalBox().getVolume(is_two_dimensions); // current volume

    unsigned int group_size = m_group->getNumMembers();

    // update degrees of freedom for MTK term
    m_ndof = m_group->getTranslationalDOF();

    // advance barostat (m_barostat.nu_xx, m_barostat.nu_yy, m_barostat.nu_zz) half a time step
    advanceBarostat(timestep);

    // Rescaling factors, including Martyna-Tobias-Klein correction
    Scalar mtk = exp(-Scalar(1.0 / 2.0) * m_deltaT
                     * (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz)
                     / static_cast<Scalar>(m_ndof));
    const auto& rf = m_thermostat ? m_thermostat->getRescalingFactorsOne(timestep, m_deltaT)
                                  : std::array<Scalar, 2> {1., 1.};
    const std::array<Scalar, 2> rescaleFactors = {rf[0] * mtk, rf[1] * mtk};

    //  update the propagator matrix using current barostat momenta
    updatePropagator();

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
            v *= rescaleFactors[0];

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
            x_zero = (I.x == 0);
            y_zero = (I.y == 0);
            z_zero = (I.z == 0);

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
            p = p * rescaleFactors[1];

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

    // propagate thermostat variables forward
    if (m_thermostat)
        {
        m_thermostat->advanceThermostat(timestep, m_deltaT, m_aniso);
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&m_barostat, 6, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    }

void TwoStepConstantPressure::thermalizeBarostatDOF(uint64_t timestep)
    {
    m_exec_conf->msg->notice(6) << "TwoStepConstantPressure randomizing barostat DOF" << std::endl;

    unsigned int instance_id = 0;
    if (m_group->getNumMembersGlobal() > 0)
        instance_id = m_group->getMemberTag(0);

    hoomd::RandomGenerator rng(
        hoomd::Seed(hoomd::RNGIdentifier::TwoStepConstantPressureThermalizeBarostat,
                    timestep,
                    m_sysdef->getSeed()),
        hoomd::Counter(instance_id));

    bool master = m_exec_conf->getRank() == 0;

    // randomize barostat variables
    unsigned int d = m_sysdef->getNDimensions();
    Scalar sigmasq_baro = Scalar(1.0) / ((Scalar)(m_ndof + d) / (Scalar)d * m_tauS * m_tauS);

    if (master)
        {
        if (m_flags & baro_x)
            {
            m_barostat.nu_xx = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_xy)
            {
            m_barostat.nu_xy = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_xz)
            {
            m_barostat.nu_xz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_y)
            {
            m_barostat.nu_yy = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_yz)
            {
            m_barostat.nu_yz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        if (m_flags & baro_z)
            {
            m_barostat.nu_zz = hoomd::NormalDistribution<Scalar>(sqrt(sigmasq_baro))(rng);
            }

        // couple box degrees of freedom
        couplingMode couple = getRelevantCouplings();

        switch (couple)
            {
        case couple_none:
            break;
        case couple_xy:
            m_barostat.nu_yy = m_barostat.nu_xx;
            break;
        case couple_xz:
            m_barostat.nu_zz = m_barostat.nu_xx;
            break;
        case couple_yz:
            m_barostat.nu_yy = m_barostat.nu_zz;
            break;
        case couple_xyz:
            m_barostat.nu_xx = m_barostat.nu_zz;
            m_barostat.nu_yy = m_barostat.nu_zz;
            break;
        default:
            throw std::runtime_error("Invalid NPT coupling mode.");
            }
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&m_barostat, 6, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    }

void TwoStepConstantPressure::integrateStepTwo(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    // Rescaling factors, including Martyna-Tobias-Klein correction
    Scalar mtk = exp(-Scalar(1.0 / 2.0) * m_deltaT
                     * (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz)
                     / static_cast<Scalar>(m_ndof));
    const auto& rf = m_thermostat ? m_thermostat->getRescalingFactorsTwo(timestep, m_deltaT)
                                  : std::array<Scalar, 2> {1., 1.};
    const std::array<Scalar, 2> rescaleFactors = {rf[0] * mtk, rf[1] * mtk};

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

        {
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                     access_location::host,
                                     access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

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
            v = v * rescaleFactors[0];

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
                x_zero = (I.x == 0);
                y_zero = (I.y == 0);
                z_zero = (I.z == 0);

                // ignore torque component along an axis for which the moment of inertia zero
                if (x_zero)
                    t.x = 0;
                if (y_zero)
                    t.y = 0;
                if (z_zero)
                    t.z = 0;

                // thermostat angular degrees of freedom
                p = p * rescaleFactors[1];

                // advance p(t+deltaT/2)->p(t+deltaT)
                p += m_deltaT * q * t;

                h_angmom.data[j] = quat_to_scalar4(p);
                }
            }
        } // end GPUArray scope

    // advance barostat (m_barostat.nu_xx, m_barostat.nu_yy, m_barostat.nu_zz) half a time step
    advanceBarostat(timestep + 1);
    }

pybind11::tuple TwoStepConstantPressure::getBarostatDOF()
    {
    return pybind11::make_tuple(m_barostat.nu_xx,
                                m_barostat.nu_xy,
                                m_barostat.nu_xz,
                                m_barostat.nu_yy,
                                m_barostat.nu_yz,
                                m_barostat.nu_zz);
    }

void TwoStepConstantPressure::setBarostatDOF(pybind11::tuple v)
    {
    if (pybind11::len(v) != 6)
        {
        throw std::length_error("barostat_dof must have length 6");
        }
    m_barostat.nu_xx = v[0].cast<Scalar>();
    m_barostat.nu_xy = v[1].cast<Scalar>();
    m_barostat.nu_xz = v[2].cast<Scalar>();
    m_barostat.nu_yy = v[3].cast<Scalar>();
    m_barostat.nu_yz = v[4].cast<Scalar>();
    m_barostat.nu_zz = v[5].cast<Scalar>();
    }

Scalar TwoStepConstantPressure::getBarostatEnergy(uint64_t timestep)
    {
    unsigned int d = m_sysdef->getNDimensions();
    Scalar W = static_cast<Scalar>(m_ndof + d) / static_cast<Scalar>(d)
               * (m_thermostat ? m_thermostat->getTimestepTemperature(timestep) : Scalar(1.0))
               * m_tauS * m_tauS;

    Scalar barostat_energy
        = W
          * (m_barostat.nu_xx * m_barostat.nu_xx + m_barostat.nu_yy * m_barostat.nu_yy
             + m_barostat.nu_zz * m_barostat.nu_zz // Normal
             + m_barostat.nu_xy * m_barostat.nu_xy + m_barostat.nu_xz * m_barostat.nu_xz
             + m_barostat.nu_yz * m_barostat.nu_yz // Shear
             )
          / Scalar(2.0);

    return barostat_energy;
    }

void TwoStepConstantPressure::updatePropagator()
    {
    // calculate some factors needed for the update matrix
    Scalar3 v_fac = make_scalar3(-Scalar(1.0 / 4.0) * m_barostat.nu_xx,
                                 -Scalar(1.0 / 4.0) * m_barostat.nu_yy,
                                 -Scalar(1.0 / 4.0) * m_barostat.nu_zz);
    Scalar3 exp_v_fac_2 = make_scalar3(exp(Scalar(2.0) * v_fac.x * m_deltaT),
                                       exp(Scalar(2.0) * v_fac.y * m_deltaT),
                                       exp(Scalar(2.0) * v_fac.z * m_deltaT));

    Scalar3 r_fac = make_scalar3(Scalar(1.0 / 2.0) * m_barostat.nu_xx,
                                 Scalar(1.0 / 2.0) * m_barostat.nu_yy,
                                 Scalar(1.0 / 2.0) * m_barostat.nu_zz);
    Scalar3 exp_r_fac = make_scalar3(exp(Scalar(1.0 / 2.0) * m_barostat.nu_xx * m_deltaT),
                                     exp(Scalar(1.0 / 2.0) * m_barostat.nu_yy * m_deltaT),
                                     exp(Scalar(1.0 / 2.0) * m_barostat.nu_zz * m_deltaT));
    Scalar3 exp_r_fac_2 = make_scalar3(exp(m_barostat.nu_xx * m_deltaT),
                                       exp(m_barostat.nu_yy * m_deltaT),
                                       exp(m_barostat.nu_zz * m_deltaT));

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
    m_mat_exp_v[0] = exp_v_fac_2.x; // xx
    m_mat_exp_v[1]
        = -m_deltaT * Scalar(1.0 / 4.0) * m_barostat.nu_xy * (exp_v_fac_2.x + exp_v_fac_2.y); // xy
    m_mat_exp_v[2]
        = -m_deltaT * Scalar(1.0 / 4.0) * m_barostat.nu_xz * (exp_v_fac_2.x + exp_v_fac_2.z)
          + m_deltaT * m_deltaT * Scalar(1.0 / 32.0) * m_barostat.nu_xy * m_barostat.nu_yz
                * (exp_v_fac_2.x + Scalar(2.0) * exp_v_fac_2.y + exp_v_fac_2.z); // xz
    m_mat_exp_v[3] = exp_v_fac_2.y;                                              // yy
    m_mat_exp_v[4]
        = -m_deltaT * Scalar(1.0 / 4.0) * m_barostat.nu_yz * (exp_v_fac_2.y + exp_v_fac_2.z); // yz
    m_mat_exp_v[5] = exp_v_fac_2.z;                                                           // zz

    // Matrix exp. for position update
    m_mat_exp_r[0] = exp_r_fac_2.x; // xx
    m_mat_exp_r[1]
        = m_deltaT * Scalar(1.0 / 2.0) * m_barostat.nu_xy * (exp_r_fac_2.x + exp_r_fac_2.y); // xy
    m_mat_exp_r[2]
        = m_deltaT * Scalar(1.0 / 2.0) * m_barostat.nu_xz * (exp_r_fac_2.x + exp_r_fac_2.z)
          + m_deltaT * m_deltaT * Scalar(1.0 / 8.0) * m_barostat.nu_xy * m_barostat.nu_yz
                * (exp_r_fac_2.x + Scalar(2.0) * exp_r_fac_2.y + exp_r_fac_2.z); // xz
    m_mat_exp_r[3] = exp_r_fac_2.y;                                              // yy
    m_mat_exp_r[4]
        = m_deltaT * Scalar(1.0 / 2.0) * m_barostat.nu_yz * (exp_r_fac_2.y + exp_r_fac_2.z); // yz
    m_mat_exp_r[5] = exp_r_fac_2.z;                                                          // zz

    // integrated matrix exp. for position update
    Scalar3 xz_fac_r = make_scalar3((Scalar(1.0) + g_r.x) * (Scalar(1.0) + g_r.x) + h_r.x,
                                    (Scalar(1.0) + g_r.y) * (Scalar(1.0) + g_r.y) + h_r.y,
                                    (Scalar(1.0) + g_r.z) * (Scalar(1.0) + g_r.z) + h_r.z);

    m_mat_exp_r_int[0] = m_deltaT * exp_r_fac.x * f_r.x; // xx
    m_mat_exp_r_int[1] = m_deltaT * m_deltaT * m_barostat.nu_xy * Scalar(1.0 / 4.0)
                         * (exp_r_fac.x * f_r.x * (Scalar(1.0) + g_r.x)
                            + exp_r_fac.y * f_r.y * (Scalar(1.0) + g_r.y)); // xy
    m_mat_exp_r_int[2]
        = m_deltaT * m_deltaT * m_barostat.nu_xz * Scalar(1.0 / 4.0)
              * (exp_r_fac.x * f_r.x * (Scalar(1.0) + g_r.x)
                 + exp_r_fac.z * f_r.z * (Scalar(1.0) + g_r.z))
          + m_deltaT * m_deltaT * m_deltaT * m_barostat.nu_xy * m_barostat.nu_yz
                * Scalar(1.0 / 32.0)
                * (exp_r_fac.x * f_r.x * xz_fac_r.x + Scalar(2.0) * exp_r_fac.y * f_r.y * xz_fac_r.y
                   + exp_r_fac.z * f_r.z * xz_fac_r.z);  // xz
    m_mat_exp_r_int[3] = m_deltaT * exp_r_fac.y * f_r.y; // yy
    m_mat_exp_r_int[4] = m_deltaT * m_deltaT * m_barostat.nu_yz * Scalar(1.0 / 4.0)
                         * (exp_r_fac.y * f_r.y * (Scalar(1.0) + g_r.y)
                            + exp_r_fac.z * f_r.z * (Scalar(1.0) + g_r.z)); // yz
    m_mat_exp_r_int[5] = m_deltaT * exp_r_fac.z * f_r.z;                    // zz
    }

void TwoStepConstantPressure::advanceBarostat(uint64_t timestep)
    {
    // compute thermodynamic properties at full time step
    m_thermo_full_step->compute(timestep);

    // compute pressure for the next half time step
    PressureTensor P = m_thermo_full_step->getPressureTensor();

    if (std::isnan(P.xx) || std::isnan(P.xy) || std::isnan(P.xz) || std::isnan(P.yy)
        || std::isnan(P.yz) || std::isnan(P.zz))
        {
        P.xx = m_S[0]->operator()(timestep);
        P.yy = m_S[1]->operator()(timestep);
        P.zz = m_S[2]->operator()(timestep);
        P.yz = m_S[3]->operator()(timestep);
        P.xz = m_S[4]->operator()(timestep);
        P.xy = m_S[5]->operator()(timestep);
        }

    // advance barostat (m_barostat.nu_xx, m_barostat.nu_yy, m_barostat.nu_zz) half a time step
    // Martyna-Tobias-Klein correction
    unsigned int d = m_sysdef->getNDimensions();
    Scalar W = static_cast<Scalar>(m_ndof + d) / static_cast<Scalar>(d)
               * (m_thermostat ? m_thermostat->getTimestepTemperature(timestep) : Scalar(1.0))
               * m_tauS * m_tauS;
    Scalar mtk_term = Scalar(2.0) * m_thermo_full_step->getTranslationalKineticEnergy();
    mtk_term *= Scalar(1.0 / 2.0) * m_deltaT / (Scalar)m_ndof / W;

    couplingMode couple = getRelevantCouplings();

    // couple diagonal elements of pressure tensor together
    Scalar3 P_diag = make_scalar3(0.0, 0.0, 0.0);
    Scalar3 R_diag = make_scalar3(0., 0., 0.);

    unsigned int instance_id = 0;
    if (m_group->getNumMembersGlobal() > 0)
        instance_id = m_group->getMemberTag(0);

    RandomGenerator rng(Seed(RNGIdentifier::ConstantPressure, timestep, m_sysdef->getSeed()),
                        instance_id);
    NormalDistribution<Scalar> noise;
    switch (couple)
        {
    case couple_none:
        P_diag.x = P.xx;
        R_diag.x = noise(rng);
        P_diag.y = P.yy;
        R_diag.y = noise(rng);
        P_diag.z = P.zz;
        R_diag.z = noise(rng);
        break;
    case couple_xy:
        P_diag.x = P_diag.y = Scalar(1.0 / 2.0) * (P.xx + P.yy);
        R_diag.x = R_diag.y = noise(rng);
        P_diag.z = P.zz;
        R_diag.z = noise(rng);
        break;
    case couple_xz:
        P_diag.x = P_diag.z = Scalar(1.0 / 2.0) * (P.xx + P.zz);
        R_diag.x = R_diag.z = noise(rng);
        P_diag.y = P.yy;
        R_diag.y = noise(rng);
        break;
    case couple_yz:
        P_diag.x = P.xx;
        R_diag.x = noise(rng);
        P_diag.y = P_diag.z = Scalar(1.0 / 2.0) * (P.yy + P.zz);
        R_diag.y = R_diag.z = noise(rng);
        break;
    case couple_xyz:
        P_diag.x = P_diag.y = P_diag.z = Scalar(1.0 / 3.0) * (P.xx + P.yy + P.zz);
        R_diag.x = R_diag.y = R_diag.z = noise(rng);
        break;
    default:
        throw std::runtime_error("Invalid NPT coupling mode.");
        }

    // update barostat matrix
    Scalar noise_exp_integrate = exp(-m_gamma * m_deltaT / Scalar(2.0));
    Scalar coeff
        = sqrt((m_thermostat ? m_thermostat->getTimestepTemperature(timestep) : Scalar(1.0))
               * (Scalar(1.0) - noise_exp_integrate * noise_exp_integrate) / W);
    if (m_flags & baro_x)
        {
        m_barostat.nu_xx = m_barostat.nu_xx * noise_exp_integrate + coeff * R_diag.x;
        m_barostat.nu_xx
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.x - m_S[0]->operator()(timestep))
               + mtk_term;
        }

    if (m_flags & baro_xy)
        {
        m_barostat.nu_xy = m_barostat.nu_xy * noise_exp_integrate + coeff * noise(rng);
        m_barostat.nu_xy
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.xy - m_S[5]->operator()(timestep));
        }

    if (m_flags & baro_xz)
        {
        m_barostat.nu_xz = m_barostat.nu_xz * noise_exp_integrate + coeff * noise(rng);
        m_barostat.nu_xz
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.xz - m_S[4]->operator()(timestep));
        }

    if (m_flags & baro_y)
        {
        m_barostat.nu_yy = m_barostat.nu_yy * noise_exp_integrate + coeff * R_diag.y;
        m_barostat.nu_yy
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.y - m_S[1]->operator()(timestep))
               + mtk_term;
        }

    if (m_flags & baro_yz)
        {
        m_barostat.nu_yz = m_barostat.nu_yz * noise_exp_integrate + coeff * noise(rng);
        m_barostat.nu_yz
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P.yz - m_S[3]->operator()(timestep));
        }

    if (m_flags & baro_z)
        {
        m_barostat.nu_zz = m_barostat.nu_zz * noise_exp_integrate + coeff * R_diag.z;
        m_barostat.nu_zz
            += Scalar(1.0 / 2.0) * m_deltaT * m_V / W * (P_diag.z - m_S[2]->operator()(timestep))
               + mtk_term;
        }
    }

namespace detail
    {
void export_TwoStepConstantPressure(pybind11::module& m)
    {
    pybind11::class_<TwoStepConstantPressure,
                     IntegrationMethodTwoStep,
                     std::shared_ptr<TwoStepConstantPressure>>(m, "TwoStepConstantPressure")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            std::vector<std::shared_ptr<Variant>>,
                            std::string,
                            std::vector<bool>,
                            std::shared_ptr<Thermostat>,
                            Scalar>())
        .def_property("tauS", &TwoStepConstantPressure::getTauS, &TwoStepConstantPressure::setTauS)
        .def_property("S", &TwoStepConstantPressure::getS, &TwoStepConstantPressure::setS)
        .def_property("couple",
                      &TwoStepConstantPressure::getCouple,
                      &TwoStepConstantPressure::setCouple)
        .def_property("box_dof",
                      &TwoStepConstantPressure::getFlags,
                      &TwoStepConstantPressure::setFlags)
        .def_property("rescale_all",
                      &TwoStepConstantPressure::getRescaleAll,
                      &TwoStepConstantPressure::setRescaleAll)
        .def_property("barostat_dof",
                      &TwoStepConstantPressure::getBarostatDOF,
                      &TwoStepConstantPressure::setBarostatDOF)
        .def("getBarostatEnergy", &TwoStepConstantPressure::getBarostatEnergy)
        .def("thermalizeBarostatDOF", &TwoStepConstantPressure::thermalizeBarostatDOF)
        .def("setThermostat", &TwoStepConstantPressure::setThermostat);
    }
    } // namespace detail

    } // namespace hoomd::md
