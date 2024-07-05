// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepLangevin.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;
using namespace hoomd;

namespace hoomd
    {
namespace md
    {
TwoStepLangevin::TwoStepLangevin(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<ParticleGroup> group,
                                 std::shared_ptr<Variant> T)
    : TwoStepLangevinBase(sysdef, group, T), m_reservoir_energy(0), m_extra_energy_overdeltaT(0),
      m_tally(false), m_noiseless_t(false), m_noiseless_r(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepLangevin" << endl;
    }

TwoStepLangevin::~TwoStepLangevin()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepLangevin" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   velocity verlet method.
*/
void TwoStepLangevin::integrateStepOne(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // perform the first half step of velocity verlet
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar dx = h_vel.data[j].x * m_deltaT
                    + Scalar(1.0 / 2.0) * h_accel.data[j].x * m_deltaT * m_deltaT;
        Scalar dy = h_vel.data[j].y * m_deltaT
                    + Scalar(1.0 / 2.0) * h_accel.data[j].y * m_deltaT * m_deltaT;
        Scalar dz = h_vel.data[j].z * m_deltaT
                    + Scalar(1.0 / 2.0) * h_accel.data[j].z * m_deltaT * m_deltaT;

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;
        // particles may have been moved slightly outside the box by the above steps, wrap them back
        // into place
        box.wrap(h_pos.data[j], h_image.data[j]);

        h_vel.data[j].x += Scalar(1.0 / 2.0) * h_accel.data[j].x * m_deltaT;
        h_vel.data[j].y += Scalar(1.0 / 2.0) * h_accel.data[j].y * m_deltaT;
        h_vel.data[j].z += Scalar(1.0 / 2.0) * h_accel.data[j].z * m_deltaT;
        }

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
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepLangevin::integrateStepTwo(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);
    ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),
                                      access_location::host,
                                      access_mode::readwrite);
    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::read);

    // grab some initial variables
    const Scalar currentTemp = m_T->operator()(timestep);
    const unsigned int D = m_sysdef->getNDimensions();

    // energy transferred over this time step
    Scalar bd_energy_transfer = 0;

    // a(t+deltaT) gets modified with the bd forces
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    uint16_t seed = m_sysdef->getSeed();

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Initialize the RNG
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepLangevin, timestep, seed),
                            hoomd::Counter(ptag));

        // first, calculate the BD forces
        // Generate three random numbers
        hoomd::UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));
        Scalar rx = uniform(rng);
        Scalar ry = uniform(rng);
        Scalar rz = uniform(rng);

        Scalar gamma;
        unsigned int type = __scalar_as_int(h_pos.data[j].w);
        gamma = h_gamma.data[type];

        // compute the bd force
        Scalar coeff = fast::sqrt(Scalar(6.0) * gamma * currentTemp / m_deltaT);
        if (m_noiseless_t)
            coeff = Scalar(0.0);
        Scalar bd_fx = rx * coeff - gamma * h_vel.data[j].x;
        Scalar bd_fy = ry * coeff - gamma * h_vel.data[j].y;
        Scalar bd_fz = rz * coeff - gamma * h_vel.data[j].z;

        if (D < 3)
            bd_fz = Scalar(0.0);

        // then, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = (h_net_force.data[j].x + bd_fx) * minv;
        h_accel.data[j].y = (h_net_force.data[j].y + bd_fy) * minv;
        h_accel.data[j].z = (h_net_force.data[j].z + bd_fz) * minv;

        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0 / 2.0) * h_accel.data[j].x * m_deltaT;
        h_vel.data[j].y += Scalar(1.0 / 2.0) * h_accel.data[j].y * m_deltaT;
        h_vel.data[j].z += Scalar(1.0 / 2.0) * h_accel.data[j].z * m_deltaT;

        // tally the energy transfer from the bd thermal reservoir to the particles
        if (m_tally)
            bd_energy_transfer
                += bd_fx * h_vel.data[j].x + bd_fy * h_vel.data[j].y + bd_fz * h_vel.data[j].z;

        // rotational updates
        if (m_aniso)
            {
            unsigned int type_r = __scalar_as_int(h_pos.data[j].w);
            Scalar3 gamma_r = h_gamma_r.data[type_r];
            // get body frame ang_mom
            quat<Scalar> p(h_angmom.data[j]);
            quat<Scalar> q(h_orientation.data[j]);
            vec3<Scalar> t(h_net_torque.data[j]);
            vec3<Scalar> I(h_inertia.data[j]);

            // s is the pure imaginary quaternion with im. part equal to true angular velocity
            vec3<Scalar> s;
            s = (Scalar(1. / 2.) * conj(q) * p).v;

            if (gamma_r.x > 0 || gamma_r.y > 0 || gamma_r.z > 0)
                {
                // first calculate in the body frame random and damping torque imposed by the
                // dynamics
                vec3<Scalar> bf_torque;

                // original Gaussian random torque
                Scalar3 sigma_r
                    = make_scalar3(fast::sqrt(Scalar(2.0) * gamma_r.x * currentTemp / m_deltaT),
                                   fast::sqrt(Scalar(2.0) * gamma_r.y * currentTemp / m_deltaT),
                                   fast::sqrt(Scalar(2.0) * gamma_r.z * currentTemp / m_deltaT));
                if (m_noiseless_r)
                    sigma_r = make_scalar3(0.0, 0.0, 0.0);

                Scalar rand_x = hoomd::NormalDistribution<Scalar>(sigma_r.x)(rng);
                Scalar rand_y = hoomd::NormalDistribution<Scalar>(sigma_r.y)(rng);
                Scalar rand_z = hoomd::NormalDistribution<Scalar>(sigma_r.z)(rng);

                // check for degenerate moment of inertia
                bool x_zero, y_zero, z_zero;
                x_zero = (I.x == 0);
                y_zero = (I.y == 0);
                z_zero = (I.z == 0);

                bf_torque.x = rand_x - gamma_r.x * (s.x / I.x);
                bf_torque.y = rand_y - gamma_r.y * (s.y / I.y);
                bf_torque.z = rand_z - gamma_r.z * (s.z / I.z);

                // ignore torque component along an axis for which the moment of inertia zero
                if (x_zero)
                    bf_torque.x = 0;
                if (y_zero)
                    bf_torque.y = 0;
                if (z_zero)
                    bf_torque.z = 0;

                // change to lab frame and update the net torque
                bf_torque = rotate(q, bf_torque);
                h_net_torque.data[j].x += bf_torque.x;
                h_net_torque.data[j].y += bf_torque.y;
                h_net_torque.data[j].z += bf_torque.z;

                if (D < 3)
                    h_net_torque.data[j].x = 0;
                if (D < 3)
                    h_net_torque.data[j].y = 0;
                }
            }
        }

    // then, update the angular velocity
    if (m_aniso)
        {
        // angular degrees of freedom
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

            // advance p(t+deltaT/2)->p(t+deltaT)
            p += m_deltaT * q * t;
            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }

    // update energy reservoir
    if (m_tally)
        {
#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &bd_energy_transfer,
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            }
#endif
        m_reservoir_energy -= bd_energy_transfer * m_deltaT;
        m_extra_energy_overdeltaT = 0.5 * bd_energy_transfer;
        }
    }

namespace detail
    {
void export_TwoStepLangevin(pybind11::module& m)
    {
    pybind11::class_<TwoStepLangevin, TwoStepLangevinBase, std::shared_ptr<TwoStepLangevin>>(
        m,
        "TwoStepLangevin")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>>())
        .def_property("tally_reservoir_energy",
                      &TwoStepLangevin::getTallyReservoirEnergy,
                      &TwoStepLangevin::setTallyReservoirEnergy)
        .def_property_readonly("reservoir_energy", &TwoStepLangevin::getReservoirEnergy);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
