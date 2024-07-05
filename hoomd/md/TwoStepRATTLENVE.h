// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegrationMethodTwoStep.h"

#ifndef __TWO_STEP_RATTLE_NVE_H__
#define __TWO_STEP_RATTLE_NVE_H__

/*! \file TwoStepRATTLENVE.h
    \brief Declares the TwoStepRATTLENVE class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/Variant.h"
#include "hoomd/VectorMath.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
constexpr unsigned int maxiteration = 10;

inline Scalar maxNorm(Scalar3 vec, Scalar resid)
    {
    Scalar vec_norm = sqrt(dot(vec, vec));
    Scalar abs_resid = fabs(resid);
    if (vec_norm > abs_resid)
        return vec_norm;
    else
        return abs_resid;
    }

//! Integrates part of the system forward in two steps in the NVE ensemble
/*! Implements velocity-verlet NVE integration through the IntegrationMethodTwoStep interface
    \ingroup updaters
*/
template<class Manifold> class PYBIND11_EXPORT TwoStepRATTLENVE : public IntegrationMethodTwoStep
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepRATTLENVE(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     Manifold manifold,
                     Scalar tolerance);

    virtual ~TwoStepRATTLENVE();

    //! Sets the movement limit
    void setLimit(std::shared_ptr<Variant>& limit)
        {
        m_limit = limit;
        }

    std::shared_ptr<Variant> getLimit()
        {
        return m_limit;
        }

    void setZeroForce(bool zero_force)
        {
        m_zero_force = zero_force;
        }

    bool getZeroForce()
        {
        return m_zero_force;
        }

    //! Performs the first step of the integration
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

    Manifold m_manifold; //!< The manifold used for the RATTLE constraint
    //!< The maximum distance a particle can move in one step
    std::shared_ptr<Variant> m_limit;
    Scalar m_tolerance; //!< The tolerance value of the RATTLE algorithm, setting the tolerance to
                        //!< the manifold
    bool m_zero_force;  //!< True if the integration step should ignore computed forces
    bool m_box_changed;
    };

/*! \file TwoStepRATTLENVE.h
    \brief Contains code for the TwoStepRATTLENVE class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param manifold The manifold describing the constraint during the RATTLE integration method
    \param tolerance Tolerance for the RATTLE iteration algorithm
*/
template<class Manifold>
TwoStepRATTLENVE<Manifold>::TwoStepRATTLENVE(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<ParticleGroup> group,
                                             Manifold manifold,
                                             Scalar tolerance)
    : IntegrationMethodTwoStep(sysdef, group), m_manifold(manifold), m_limit(),
      m_tolerance(tolerance), m_zero_force(false), m_box_changed(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepRATTLENVE" << std::endl;

    m_pdata->getBoxChangeSignal()
        .template connect<TwoStepRATTLENVE<Manifold>, &TwoStepRATTLENVE<Manifold>::setBoxChange>(
            this);

    if (!m_manifold.fitsInsideBox(m_pdata->getGlobalBox()))
        {
        throw std::runtime_error("Parts of the manifold are outside the box");
        }
    }

template<class Manifold> TwoStepRATTLENVE<Manifold>::~TwoStepRATTLENVE()
    {
    m_pdata->getBoxChangeSignal()
        .template disconnect<TwoStepRATTLENVE<Manifold>, &TwoStepRATTLENVE<Manifold>::setBoxChange>(
            this);
    m_exec_conf->msg->notice(5) << "Destroying TwoStepRATTLENVE" << std::endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   velocity verlet method.
*/
template<class Manifold> void TwoStepRATTLENVE<Manifold>::integrateStepOne(uint64_t timestep)
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
    // v(t+deltaT/2) = v(t) + (1/2)*deltaT*(a-lambda*n_manifold(x(t))/m)
    // iterative: x(t+deltaT) = x(t+deltaT) - J^(-1)*residual
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        if (m_zero_force)
            {
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;
            }

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

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar maximum_displacement = m_limit->operator()(timestep);
            Scalar len = sqrt(dx * dx + dy * dy + dz * dz);
            if (len > maximum_displacement)
                {
                dx = dx / len * maximum_displacement;
                dy = dy / len * maximum_displacement;
                dz = dz / len * maximum_displacement;
                }
            }

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;
        }

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        box.wrap(h_pos.data[j], h_image.data[j]);
        }

    // Integration of angular degrees of freedom using symplectic and
    // time-reversal symmetric integration scheme of Miller et al.
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
template<class Manifold> void TwoStepRATTLENVE<Manifold>::integrateStepTwo(uint64_t timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(),
                                 access_location::host,
                                 access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    // iterative: v(t+deltaT) = v(t+deltaT/2) - J^(-1)*residual
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar mass = h_vel.data[j].w;
        Scalar inv_mass = Scalar(1.0) / mass;

        if (m_zero_force)
            {
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;
            }
        else
            {
            // first, calculate acceleration from the net force
            h_accel.data[j].x = h_net_force.data[j].x * inv_mass;
            h_accel.data[j].y = h_net_force.data[j].y * inv_mass;
            h_accel.data[j].z = h_net_force.data[j].z * inv_mass;
            }

        Scalar mu = 0;
        Scalar inv_alpha = -Scalar(1.0 / 2.0) * m_deltaT;
        inv_alpha = Scalar(1.0) / inv_alpha;

        Scalar3 normal = m_manifold.derivative(
            make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z));

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

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar maximum_displacement = m_limit->operator()(timestep);
            Scalar vel = sqrt(h_vel.data[j].x * h_vel.data[j].x + h_vel.data[j].y * h_vel.data[j].y
                              + h_vel.data[j].z * h_vel.data[j].z);
            if ((vel * m_deltaT) > maximum_displacement)
                {
                h_vel.data[j].x = h_vel.data[j].x / vel * maximum_displacement / m_deltaT;
                h_vel.data[j].y = h_vel.data[j].y / vel * maximum_displacement / m_deltaT;
                h_vel.data[j].z = h_vel.data[j].z / vel * maximum_displacement / m_deltaT;
                }
            }
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

            // advance p(t+deltaT/2)->p(t+deltaT)
            p += m_deltaT * q * t;

            h_angmom.data[j] = quat_to_scalar4(p);
            }
        }
    }

template<class Manifold> void TwoStepRATTLENVE<Manifold>::includeRATTLEForce(uint64_t timestep)
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
    // v(t+deltaT/2) = v(t) + (1/2)*deltaT*(a-lambda*n_manifold(x(t))/m)
    // iterative: x(t+deltaT) = x(t+deltaT) - J^(-1)*residual
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        if (m_zero_force)
            {
            h_accel.data[j].x = h_accel.data[j].y = h_accel.data[j].z = 0.0;
            }

        Scalar lambda = 0.0;

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

        unsigned int iteration = 0;
        do
            {
            iteration++;
            half_vel.x = h_vel.data[j].x
                         + deltaT_half * (h_accel.data[j].x - inv_mass * lambda * normal.x);
            half_vel.y = h_vel.data[j].y
                         + deltaT_half * (h_accel.data[j].y - inv_mass * lambda * normal.y);
            half_vel.z = h_vel.data[j].z
                         + deltaT_half * (h_accel.data[j].z - inv_mass * lambda * normal.z);

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
            lambda = lambda - beta * inv_alpha;

            } while (maxNorm(residual, resid) > m_tolerance && iteration < maxiteration);

        if (iteration == maxiteration)
            {
            m_exec_conf->msg->warning()
                << "The RATTLE integrator needed an unusual high number of iterations!" << std::endl
                << "It is recomended to change the initial configuration or lower the step size."
                << std::endl;
            }

        h_net_force.data[j].x -= lambda * normal.x;
        h_net_force.data[j].y -= lambda * normal.y;
        h_net_force.data[j].z -= lambda * normal.z;

        h_net_virial.data[0 * net_virial_pitch + j] -= lambda * normal.x * h_pos.data[j].x;
        h_net_virial.data[1 * net_virial_pitch + j]
            -= 0.5 * lambda * (normal.y * h_pos.data[j].x + normal.x * h_pos.data[j].y);
        h_net_virial.data[2 * net_virial_pitch + j]
            -= 0.5 * lambda * (normal.z * h_pos.data[j].x + normal.x * h_pos.data[j].z);
        h_net_virial.data[3 * net_virial_pitch + j] -= lambda * normal.y * h_pos.data[j].y;
        h_net_virial.data[4 * net_virial_pitch + j]
            -= 0.5 * lambda * (normal.y * h_pos.data[j].z + normal.z * h_pos.data[j].y);
        h_net_virial.data[5 * net_virial_pitch + j] -= lambda * normal.z * h_pos.data[j].z;

        h_accel.data[j].x -= inv_mass * lambda * normal.x;
        h_accel.data[j].y -= inv_mass * lambda * normal.y;
        h_accel.data[j].z -= inv_mass * lambda * normal.z;
        }
    }

namespace detail
    {
template<class Manifold> void export_TwoStepRATTLENVE(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<TwoStepRATTLENVE<Manifold>,
                     IntegrationMethodTwoStep,
                     std::shared_ptr<TwoStepRATTLENVE<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold,
                            Scalar>())
        .def_property("maximum_displacement",
                      &TwoStepRATTLENVE<Manifold>::getLimit,
                      &TwoStepRATTLENVE<Manifold>::setLimit)
        .def_property("zero_force",
                      &TwoStepRATTLENVE<Manifold>::getZeroForce,
                      &TwoStepRATTLENVE<Manifold>::setZeroForce)
        .def_property("tolerance",
                      &TwoStepRATTLENVE<Manifold>::getTolerance,
                      &TwoStepRATTLENVE<Manifold>::setTolerance);
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __TWO_STEP_RATTLENVE_H__
