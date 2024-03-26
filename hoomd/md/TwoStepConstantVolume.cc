// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepConstantVolume.h"
#include "hoomd/VectorMath.h"

void hoomd::md::TwoStepConstantVolume::integrateStepOne(uint64_t timestep)
    {
    if (m_group->getNumMembersGlobal() == 0)
        {
        throw std::runtime_error("Empty integration group.");
        }

    auto rescaling_factors = m_thermostat ? m_thermostat->getRescalingFactorsOne(timestep, m_deltaT)
                                          : std::array<Scalar, 2> {1., 1.};

    unsigned int group_size = m_group->getNumMembers();

        // scope array handles for proper releasing before calling the thermo compute
        {
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                     access_location::host,
                                     access_mode::readwrite);
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            // load variables
            Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
            Scalar3 pos = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
            Scalar3 accel = h_accel.data[j];

            // update velocity and position
            v = v + Scalar(1.0 / 2.0) * accel * m_deltaT;

            // rescale velocity
            v *= rescaling_factors[0];
            if (m_limit)
                {
                auto maximum_displacement = m_limit->operator()(timestep);
                auto len = sqrt(dot(v, v)) * m_deltaT;
                if (len > maximum_displacement)
                    {
                    v = v / len * maximum_displacement / m_deltaT;
                    }
                }
            pos += m_deltaT * v;

            // store updated variables
            h_vel.data[j].x = v.x;
            h_vel.data[j].y = v.y;
            h_vel.data[j].z = v.z;

            h_pos.data[j].x = pos.x;
            h_pos.data[j].y = pos.y;
            h_pos.data[j].z = pos.z;
            }

        // particles may have been moved slightly outside the box by the above steps, wrap them back
        // into place
        const BoxDim& box = m_pdata->getBox();

        ArrayHandle<int3> h_image(m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            // wrap the particles around the box
            box.wrap(h_pos.data[j], h_image.data[j]);
            }
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
            // using Trotter factorization of rotation Liouvillian
            p += m_deltaT * q * t;

            // apply thermostat
            p = p * rescaling_factors[1];

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

    // get temperature and advance thermostat
    if (m_thermostat)
        {
        m_thermostat->advanceThermostat(timestep, m_deltaT, m_aniso);
        }
    }

void hoomd::md::TwoStepConstantVolume::integrateStepTwo(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    auto rescaling_factors = m_thermostat ? m_thermostat->getRescalingFactorsTwo(timestep, m_deltaT)
                                          : std::array<Scalar, 2> {1., 1.};

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // perform second half step of Nose-Hoover integration

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // load velocity
        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        Scalar3 accel = h_accel.data[j];
        Scalar3 net_force
            = make_scalar3(h_net_force.data[j].x, h_net_force.data[j].y, h_net_force.data[j].z);

        // first, calculate acceleration from the net force
        Scalar m = h_vel.data[j].w;
        Scalar minv = Scalar(1.0) / m;
        accel = net_force * minv;

        // rescale velocity
        v *= rescaling_factors[0];

        // update velocity
        v += Scalar(1.0 / 2.0) * m_deltaT * accel;

        // store velocity
        h_vel.data[j].x = v.x;
        h_vel.data[j].y = v.y;
        h_vel.data[j].z = v.z;

        // store acceleration
        h_accel.data[j] = accel;
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

            // apply thermostat
            p = p * rescaling_factors[1];

            // advance p(t+deltaT/2)->p(t+deltaT)
            p += m_deltaT * q * t;

            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }
    }

namespace hoomd::md::detail
    {
void export_TwoStepConstantVolume(pybind11::module& m)
    {
    pybind11::class_<TwoStepConstantVolume,
                     IntegrationMethodTwoStep,
                     std::shared_ptr<TwoStepConstantVolume>>(m, "TwoStepConstantVolume")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Thermostat>>())
        .def("setThermostat", &TwoStepConstantVolume::setThermostat)
        .def_property("maximum_displacement",
                      &TwoStepConstantVolume::getLimit,
                      &TwoStepConstantVolume::setLimit);
    }
    }; // namespace hoomd::md::detail
