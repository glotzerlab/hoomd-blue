// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepLangevinBase.h"
#include "TwoStepRATTLENVE.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/VectorMath.h"

#ifndef __TWO_STEP_RATTLE_LANGEVIN_H__
#define __TWO_STEP_RATTLE_LANGEVIN_H__

/*! \file TwoStepRATTLELangevin.h
    \brief Declares the TwoStepLangevin class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Integrates part of the system forward in two steps with Langevin dynamics
/*! Implements Langevin dynamics.

    Langevin dynamics modifies standard NVE integration with two additional forces, a random force
   and a drag force. This implementation is very similar to TwoStepNVE with the additional forces.
   Note that this is not a really proper Langevin integrator, but it works well in practice.

    \ingroup updaters
*/
template<class Manifold> class PYBIND11_EXPORT TwoStepRATTLELangevin : public TwoStepLangevinBase
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepRATTLELangevin(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<ParticleGroup> group,
                          Manifold manifold,
                          std::shared_ptr<Variant> T,
                          Scalar tolerance = 0.000001);
    virtual ~TwoStepRATTLELangevin();

    void setTallyReservoirEnergy(bool tally)
        {
        m_tally = tally;
        }

    /// Get the tally setting
    bool getTallyReservoirEnergy()
        {
        return m_tally;
        }

    /// Get the reservoir energy
    Scalar getReservoirEnergy()
        {
        return m_reservoir_energy;
        }

    //! Performs the second step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    //! Includes the RATTLE forces to the virial/net force
    virtual void includeRATTLEForce(uint64_t timestep);

    //! Get the number of degrees of freedom granted to a given group
    virtual Scalar getTranslationalDOF(std::shared_ptr<ParticleGroup> group)
        {
        // get the size of the intersection between query_group and m_group
        unsigned int intersect_size
            = ParticleGroup::groupIntersection(group, m_group)->getNumMembersGlobal();

        return Manifold::dimension() * intersect_size;
        }

    /// Sets tolerance
    void setTolerance(Scalar tolerance)
        {
        m_tolerance = tolerance;
        };

    /// Gets tolerance
    Scalar getTolerance()
        {
        return m_tolerance;
        };

    protected:
    //! Helper function to be called when box changes
    void setBoxChange()
        {
        m_box_changed = true;
        }

    Manifold m_manifold;       //!< The manifold used for the RATTLE constraint
    Scalar m_reservoir_energy; //!< The energy of the reservoir the system is coupled to.
    Scalar
        m_extra_energy_overdeltaT; //!< An energy packet that isn't added until the next time step
    bool m_tally;       //!< If true, changes to the energy of the reservoir are calculated
    bool m_noiseless_t; //!< If set true, there will be no translational noise (random force)
    bool m_noiseless_r; //!< If set true, there will be no rotational noise (random torque)
    Scalar m_tolerance; //!< The tolerance value of the RATTLE algorithm, setting the tolerance to
                        //!< the manifold
    bool m_box_changed;
    };

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param manifold The manifold describing the constraint during the RATTLE integration method
    \param T Temperature set point as a function of time
    \param noiseless_t If set true, there will be no translational noise (random force)
    \param noiseless_r If set true, there will be no rotational noise (random torque)
    \param tolerance Tolerance for the RATTLE iteration algorithm

*/
template<class Manifold>
TwoStepRATTLELangevin<Manifold>::TwoStepRATTLELangevin(std::shared_ptr<SystemDefinition> sysdef,
                                                       std::shared_ptr<ParticleGroup> group,
                                                       Manifold manifold,
                                                       std::shared_ptr<Variant> T,
                                                       Scalar tolerance)
    : TwoStepLangevinBase(sysdef, group, T), m_manifold(manifold), m_reservoir_energy(0),
      m_extra_energy_overdeltaT(0), m_tally(false), m_noiseless_t(false), m_noiseless_r(false),
      m_tolerance(tolerance), m_box_changed(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepRATTLELangevin" << std::endl;

    m_pdata->getBoxChangeSignal()
        .template connect<TwoStepRATTLELangevin<Manifold>,
                          &TwoStepRATTLELangevin<Manifold>::setBoxChange>(this);

    if (!m_manifold.fitsInsideBox(m_pdata->getGlobalBox()))
        {
        throw std::runtime_error("Parts of the manifold are outside the box");
        }
    }

template<class Manifold> TwoStepRATTLELangevin<Manifold>::~TwoStepRATTLELangevin()
    {
    m_pdata->getBoxChangeSignal()
        .template disconnect<TwoStepRATTLELangevin<Manifold>,
                             &TwoStepRATTLELangevin<Manifold>::setBoxChange>(this);
    m_exec_conf->msg->notice(5) << "Destroying TwoStepRATTLELangevin" << std::endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   velocity verlet method.
*/
template<class Manifold> void TwoStepRATTLELangevin<Manifold>::integrateStepOne(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar3> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);

    const BoxDim box = m_pdata->getBox();

    if (m_box_changed)
        {
        if (!m_manifold.fitsInsideBox(m_pdata->getGlobalBox()))
            {
            throw std::runtime_error("Parts of the manifold are outside the box");
            }
        m_box_changed = false;
        }

    // perform the first half step of the RATTLE algorithm applied on velocity verlet
    // v(t+deltaT/2) = v(t) + (1/2)*deltaT*(a-alpha*n_manifold(x(t))/m)
    // iterative: x(t+deltaT) = x(t+deltaT) - J^(-1)*residual
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar deltaT_half = Scalar(1.0 / 2.0) * m_deltaT;

        Scalar3 half_vel;
        half_vel.x = h_vel.data[j].x + deltaT_half * h_accel.data[j].x;
        half_vel.y = h_vel.data[j].y + deltaT_half * h_accel.data[j].y;
        half_vel.z = h_vel.data[j].z + deltaT_half * h_accel.data[j].z;

        h_vel.data[j].x = half_vel.x;
        h_vel.data[j].y = half_vel.y;
        h_vel.data[j].z = half_vel.z;

        Scalar dx = m_deltaT * half_vel.x;
        Scalar dy = m_deltaT * half_vel.y;
        Scalar dz = m_deltaT * half_vel.z;

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;

        // particles may have been moved slightly outside the box by the above steps, wrap them back
        // into place
        box.wrap(h_pos.data[j], h_image.data[j]);
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
template<class Manifold> void TwoStepRATTLELangevin<Manifold>::integrateStepTwo(uint64_t timestep)
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

    // energy transferred over this time step
    Scalar bd_energy_transfer = 0;

    uint16_t seed = m_sysdef->getSeed();

    // a(t+deltaT) gets modified with the bd forces
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    // iterative: v(t+deltaT) = v(t+deltaT/2) - J^(-1)*residual
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Initialize the RNG
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::TwoStepLangevin, timestep, seed),
                            hoomd::Counter(ptag));

        // first, calculate the BD forces on manifold
        // Generate two random numbers

        Scalar rx, ry, rz, coeff;

        Scalar gamma;
        unsigned int type = __scalar_as_int(h_pos.data[j].w);
        gamma = h_gamma.data[type];

        Scalar3 normal = m_manifold.derivative(
            make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z));
        Scalar ndotn = dot(normal, normal);

        if (currentTemp > 0)
            {
            hoomd::UniformDistribution<Scalar> uniform(Scalar(-1), Scalar(1));

            rx = uniform(rng);
            ry = uniform(rng);
            rz = uniform(rng);

            // compute the bd force
            coeff = fast::sqrt(Scalar(6.0) * gamma * currentTemp / m_deltaT);
            if (m_noiseless_t)
                coeff = Scalar(0.0);

            Scalar proj_x = normal.x / fast::sqrt(ndotn);
            Scalar proj_y = normal.y / fast::sqrt(ndotn);
            Scalar proj_z = normal.z / fast::sqrt(ndotn);

            Scalar proj_r = rx * proj_x + ry * proj_y + rz * proj_z;
            rx = rx - proj_r * proj_x;
            ry = ry - proj_r * proj_y;
            rz = rz - proj_r * proj_z;
            }
        else
            {
            rx = 0;
            ry = 0;
            rz = 0;
            coeff = 0;
            }

        Scalar bd_fx = rx * coeff - gamma * h_vel.data[j].x;
        Scalar bd_fy = ry * coeff - gamma * h_vel.data[j].y;
        Scalar bd_fz = rz * coeff - gamma * h_vel.data[j].z;

        // then, calculate acceleration from the net force
        Scalar mass = h_vel.data[j].w;
        Scalar inv_mass = Scalar(1.0) / mass;
        h_accel.data[j].x = (h_net_force.data[j].x + bd_fx) * inv_mass;
        h_accel.data[j].y = (h_net_force.data[j].y + bd_fy) * inv_mass;
        h_accel.data[j].z = (h_net_force.data[j].z + bd_fz) * inv_mass;

        Scalar mu = 0;
        Scalar inv_alpha = -Scalar(1.0 / 2.0) * m_deltaT;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 next_vel;
        next_vel.x = h_vel.data[j].x + Scalar(1.0 / 2.0) * m_deltaT * h_accel.data[j].x;
        next_vel.y = h_vel.data[j].y + Scalar(1.0 / 2.0) * m_deltaT * h_accel.data[j].y;
        next_vel.z = h_vel.data[j].z + Scalar(1.0 / 2.0) * m_deltaT * h_accel.data[j].z;

        Scalar3 residual;
        Scalar resid;
        Scalar3 vel_dot;

        unsigned int iteration = 0;
        do
            {
            iteration++;
            vel_dot.x = h_accel.data[j].x - mu * inv_mass * normal.x;
            vel_dot.y = h_accel.data[j].y - mu * inv_mass * normal.y;
            vel_dot.z = h_accel.data[j].z - mu * inv_mass * normal.z;

            residual.x = h_vel.data[j].x - next_vel.x + Scalar(1.0 / 2.0) * m_deltaT * vel_dot.x;
            residual.y = h_vel.data[j].y - next_vel.y + Scalar(1.0 / 2.0) * m_deltaT * vel_dot.y;
            residual.z = h_vel.data[j].z - next_vel.z + Scalar(1.0 / 2.0) * m_deltaT * vel_dot.z;
            resid = dot(normal, next_vel) * inv_mass;

            Scalar ndotr = dot(normal, residual);
            Scalar ndotn = dot(normal, normal);
            Scalar beta = (mass * resid + ndotr) / ndotn;
            next_vel.x = next_vel.x - normal.x * beta + residual.x;
            next_vel.y = next_vel.y - normal.y * beta + residual.y;
            next_vel.z = next_vel.z - normal.z * beta + residual.z;
            mu = mu - mass * beta * inv_alpha;

            } while (maxNorm(residual, resid) * mass > m_tolerance && iteration < maxiteration);

        if (iteration == maxiteration)
            {
            m_exec_conf->msg->warning()
                << "The RATTLE integrator needed an unusual high number of iterations!" << std::endl
                << "It is recomended to change the initial configuration or lower the step size."
                << std::endl;
            }

        // then, update the velocity
        h_vel.data[j].x
            += Scalar(1.0 / 2.0) * m_deltaT * (h_accel.data[j].x - mu * inv_mass * normal.x);
        h_vel.data[j].y
            += Scalar(1.0 / 2.0) * m_deltaT * (h_accel.data[j].y - mu * inv_mass * normal.y);
        h_vel.data[j].z
            += Scalar(1.0 / 2.0) * m_deltaT * (h_accel.data[j].z - mu * inv_mass * normal.z);

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

template<class Manifold> void TwoStepRATTLELangevin<Manifold>::includeRATTLEForce(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::readwrite);

    size_t net_virial_pitch = net_virial.getPitch();

    // perform the first half step of the RATTLE algorithm applied on velocity verlet
    // v(t+deltaT/2) = v(t) + (1/2)*deltaT*(a-alpha*n_manifold(x(t))/m)
    // iterative: x(t+deltaT) = x(t+deltaT) - J^(-1)*residual
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar alpha = 0.0;

        Scalar3 next_pos;
        next_pos.x = h_pos.data[j].x;
        next_pos.y = h_pos.data[j].y;
        next_pos.z = h_pos.data[j].z;

        Scalar3 normal = m_manifold.derivative(next_pos);

        Scalar inv_mass = Scalar(1.0) / h_vel.data[j].w;
        Scalar deltaT_half = Scalar(1.0 / 2.0) * m_deltaT;
        Scalar inv_alpha = -deltaT_half * m_deltaT * inv_mass;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 residual;
        Scalar resid;
        Scalar3 half_vel;

        unsigned int maxiteration = 10;
        unsigned int iteration = 0;
        do
            {
            iteration++;
            half_vel.x
                = h_vel.data[j].x + deltaT_half * (h_accel.data[j].x - inv_mass * alpha * normal.x);
            half_vel.y
                = h_vel.data[j].y + deltaT_half * (h_accel.data[j].y - inv_mass * alpha * normal.y);
            half_vel.z
                = h_vel.data[j].z + deltaT_half * (h_accel.data[j].z - inv_mass * alpha * normal.z);

            residual.x = h_pos.data[j].x - next_pos.x + m_deltaT * half_vel.x;
            residual.y = h_pos.data[j].y - next_pos.y + m_deltaT * half_vel.y;
            residual.z = h_pos.data[j].z - next_pos.z + m_deltaT * half_vel.z;
            resid = m_manifold.implicitFunction(next_pos);

            Scalar3 next_normal = m_manifold.derivative(next_pos);
            Scalar nndotr = dot(next_normal, residual);
            Scalar nndotn = dot(next_normal, normal);
            Scalar beta = (resid + nndotr) / nndotn;

            next_pos.x = next_pos.x - beta * normal.x + residual.x;
            next_pos.y = next_pos.y - beta * normal.y + residual.y;
            next_pos.z = next_pos.z - beta * normal.z + residual.z;
            alpha = alpha - beta * inv_alpha;

            } while (maxNorm(residual, resid) > m_tolerance && iteration < maxiteration);

        if (iteration == maxiteration)
            {
            m_exec_conf->msg->warning()
                << "The RATTLE integrator needed an unusual high number of iterations!" << std::endl
                << "It is recomended to change the initial configuration or lower the step size."
                << std::endl;
            }

        h_net_force.data[j].x -= alpha * normal.x;
        h_net_force.data[j].y -= alpha * normal.y;
        h_net_force.data[j].z -= alpha * normal.z;

        h_net_virial.data[0 * net_virial_pitch + j] -= alpha * normal.x * h_pos.data[j].x;
        h_net_virial.data[1 * net_virial_pitch + j]
            -= 0.5 * alpha * (normal.y * h_pos.data[j].x + normal.x * h_pos.data[j].y);
        h_net_virial.data[2 * net_virial_pitch + j]
            -= 0.5 * alpha * (normal.z * h_pos.data[j].x + normal.x * h_pos.data[j].z);
        h_net_virial.data[3 * net_virial_pitch + j] -= alpha * normal.y * h_pos.data[j].y;
        h_net_virial.data[4 * net_virial_pitch + j]
            -= 0.5 * alpha * (normal.y * h_pos.data[j].z + normal.z * h_pos.data[j].y);
        h_net_virial.data[5 * net_virial_pitch + j] -= alpha * normal.z * h_pos.data[j].z;

        h_accel.data[j].x -= inv_mass * alpha * normal.x;
        h_accel.data[j].y -= inv_mass * alpha * normal.y;
        h_accel.data[j].z -= inv_mass * alpha * normal.z;
        }
    }

namespace detail
    {
template<class Manifold>
void export_TwoStepRATTLELangevin(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<TwoStepRATTLELangevin<Manifold>,
                     TwoStepLangevinBase,
                     std::shared_ptr<TwoStepRATTLELangevin<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold,
                            std::shared_ptr<Variant>,
                            Scalar>())
        .def_property("tally_reservoir_energy",
                      &TwoStepRATTLELangevin<Manifold>::getTallyReservoirEnergy,
                      &TwoStepRATTLELangevin<Manifold>::setTallyReservoirEnergy)
        .def_property("tolerance",
                      &TwoStepRATTLELangevin<Manifold>::getTolerance,
                      &TwoStepRATTLELangevin<Manifold>::setTolerance);
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __TWO_STEP_RATTLE_LANGEVIN_H__
