//
// Created by girard01 on 10/26/22.
//

#include "TwoStepNPTBase.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

namespace hoomd::md
    {

TwoStepNPTBase::TwoStepNPTBase(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<ParticleGroup> group,
                               std::shared_ptr<ComputeThermo> thermo_half_step,
                               std::shared_ptr<ComputeThermo> thermo_full_step,
                               std::shared_ptr<Variant> T,
                               const std::vector<std::shared_ptr<Variant>>& S,
                               const std::string& couple,
                               const std::vector<bool>& flags,
                               const bool nph) :
      IntegrationMethodTwoStep(sysdef, group),
      m_thermo_half_step(thermo_half_step),
      m_thermo_full_step(thermo_full_step),
      m_T(T),
      m_S(S),
      m_nph(nph),
      m_ndof(0),
      m_rescale_all(false)
    {

    if (m_flags == 0)
        m_exec_conf->msg->warning() << "integrate.npt: No barostat couplings specified." << std::endl;

    setCouple(couple);
    setFlags(flags);

    bool is_two_dimensions = m_sysdef->getNDimensions() == 2;
    m_V = m_pdata->getGlobalBox().getVolume(is_two_dimensions); // volume
    }

void TwoStepNPTBase::setCouple(const std::string& value)
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

std::string TwoStepNPTBase::getCouple()
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

TwoStepNPTBase::couplingMode TwoStepNPTBase::getRelevantCouplings()
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
void TwoStepNPTBase::setFlags(const std::vector<bool>& value)
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
std::vector<bool> TwoStepNPTBase::getFlags()
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

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   Martyna-Tobias-Klein barostat and thermostat
*/
void TwoStepNPTBase::integrateStepOne(uint64_t timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        throw std::runtime_error("Invalid NPT coupling mode.");
        }

    // update box dimensions
    bool is_two_dimensions = m_sysdef->getNDimensions() == 2;

    m_V = m_pdata->getGlobalBox().getVolume(is_two_dimensions); // current volume

    unsigned int group_size = m_group->getNumMembers();

    const auto&& rescaleFactors = NPT_thermo_rescale_factor_one(timestep);

    // update degrees of freedom for MTK term
    m_ndof = m_group->getTranslationalDOF();

    // advance barostat (m_barostat.nu_xx, m_barostat.nu_yy, m_barostat.nu_zz) half a time step
    advanceBarostat(timestep);

    // Martyna-Tobias-Klein correction
    //Scalar mtk = (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz) / (Scalar)m_ndof;

    // update the propagator matrix using current barostat momenta
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
        //Scalar xi_trans = m_thermostat.xi;
        //Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (xi_trans + mtk) * m_deltaT);

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
            v *= rescaleFactors[0];// exp_thermo_fac;

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
        //const Scalar xi_rot = m_thermostat.xi_rot;
        //Scalar exp_thermo_fac_rot = exp(-(xi_rot + mtk) * m_deltaT / Scalar(2.0));

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
            p = p * rescaleFactors[1]; //exp_thermo_fac_rot;

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
        MPI_Bcast(&m_thermostat, 4, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&m_barostat, 6, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    }


/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNPTBase::integrateStepTwo(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    const auto&& rescaleFactors = NPT_thermo_rescale_factor_two(timestep);

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

        {
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                     access_location::host,
                                     access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

        // precompute loop invariant quantity
        //Scalar xi_trans = m_thermostat.xi;
       // Scalar mtk = (m_barostat.nu_xx + m_barostat.nu_yy + m_barostat.nu_zz) / (Scalar)m_ndof;
        //Scalar exp_thermo_fac = exp(-Scalar(1.0 / 2.0) * (xi_trans + mtk) * m_deltaT);

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
            v = v * rescaleFactors[0];// exp_thermo_fac;

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
            //const Scalar xi_rot = m_thermostat.xi_rot;
            //Scalar exp_thermo_fac_rot = exp(-(xi_rot + mtk) * m_deltaT / Scalar(2.0));

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
                p = p * rescaleFactors[1]; //exp_thermo_fac_rot;

                // advance p(t+deltaT/2)->p(t+deltaT)
                p += m_deltaT * q * t;

                h_angmom.data[j] = quat_to_scalar4(p);
                }
            }
        } // end GPUArray scope

    // advance barostat (m_barostat.nu_xx, m_barostat.nu_yy, m_barostat.nu_zz) half a time step
    advanceBarostat(timestep + 1);
    }


namespace detail{
void export_TwoStepNPTBase(pybind11::module& m){
    pybind11::class_<TwoStepNPTBase, IntegrationMethodTwoStep, std::shared_ptr<TwoStepNPTBase>>(m, "TwoStepNPTBase")
    .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>,
        std::shared_ptr<ComputeThermo>, std::shared_ptr<ComputeThermo>, std::shared_ptr<Variant>,
            std::vector<std::shared_ptr<Variant>>, std::string, std::vector<bool>, bool>())
        .def_property("kT", &TwoStepNPTBase::getT, &TwoStepNPTBase::setT)
        .def_property("S", &TwoStepNPTBase::getS, &TwoStepNPTBase::setS)
        .def_property("couple", &TwoStepNPTBase::getCouple, &TwoStepNPTBase::setCouple)
        .def_property("box_dof", &TwoStepNPTBase::getFlags, &TwoStepNPTBase::setFlags)
        .def_property("rescale_all", &TwoStepNPTBase::getRescaleAll, &TwoStepNPTBase::setRescaleAll)
        ;
    }
    }

    }