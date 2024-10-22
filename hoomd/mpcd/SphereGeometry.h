// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SphereGeometry.h
 * \brief Definition of the MPCD sphere geometry
 */

#ifndef MPCD_SPHERE_GEOMETRY_H_
#define MPCD_SPHERE_GEOMETRY_H_

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__ inline
#else
#define HOSTDEVICE inline __attribute__((always_inline))
#include <string>
#endif // __HIPCC__

namespace hoomd
    {
namespace mpcd
    {
//! Sphere geometry
/*!
 * This models a fluid confined inside a sphere, centered at the origin with radius \a R.
 * If a particle leaves the sphere in a single simulation step, the particle is backtracked to the
 * point on the surface from which it exited the surface and then reflected according to appropriate
 * boundary condition.
 */
class __attribute__((visibility("default"))) SphereGeometry
    {
    public:
    //! Constructor
    /*!
     * \param R Radius
     * \param no_slip Boundary condition at the wall (slip or no-slip)
     */
    HOSTDEVICE SphereGeometry(Scalar R, bool no_slip) : m_R2(R * R), m_no_slip(no_slip) { }

    //! Detect collision between the particle and the boundary
    /*!
     * \param pos Proposed particle position
     * \param vel Proposed particle velocity
     * \param dt Integration time remaining
     *
     * \returns True if a collision occurred, and false otherwise
     *
     * \post The particle position \a pos is moved to the point of reflection, the velocity \a vel
     * is updated according to the appropriate bounce back rule, and the integration time \a dt is
     * decreased to the amount of time remaining.
     */
    HOSTDEVICE bool detectCollision(Scalar3& pos, Scalar3& vel, Scalar& dt) const
        {
        /*
         * If particle is still inside the sphere or stationary, no collision could have occurred
         * and therefore exit immediately. If particle is on surface, we are assuming it's still
         * inside and if it goes outside (during next streaming step) we can backtrack it in the end
         * of next streaming step.
         */
        const Scalar r2 = dot(pos, pos);
        const Scalar v2 = dot(vel, vel);
        if (r2 <= m_R2 || v2 == Scalar(0.0))
            {
            dt = Scalar(0);
            return false;
            }

        /*
         * Find the time remaining when the particle collided with the sphere of radius R. This time
         * is found by backtracking the position, r* = r-dt*v, and solving for dt when dot(r*,r*) =
         * R^2. This gives a quadratic equation in dt; the smaller root is the solution.
         */
        const Scalar rv = dot(pos, vel);
        dt = (rv - slow::sqrt(rv * rv - v2 * (r2 - m_R2))) / v2;

        // backtrack the particle for time dt to get to point of contact
        pos -= vel * dt;

        // update velocity according to boundary conditions
        if (m_no_slip)
            {
            vel = -vel;
            }
        else
            {
            // only no-penetration condition is enforced, so only v_perp is reflected.
            const Scalar3 vperp = dot(vel, pos) * pos / m_R2;
            vel -= Scalar(2) * vperp;
            }

        return true;
        }

    //! Check if a particle is out of bounds
    /*!
     * \param pos Current particle position
     * \returns True if particle is out of bounds, and false otherwise
     */
    HOSTDEVICE bool isOutside(const Scalar3& pos) const
        {
        return dot(pos, pos) > m_R2;
        }

    //! Add a contribution to random virtual particle velocity.
    /*!
     * \param vel Velocity of virtual particle
     * \param pos Position of virtual particle
     *
     * No velocity contribution is needed as the wall is stationary.
     */
    HOSTDEVICE void addToVirtualParticleVelocity(Scalar3& vel, const Scalar3& pos) const { }

    //! Get Sphere radius
    /*!
     * \returns confinement radius
     */
    HOSTDEVICE Scalar getRadius() const
        {
        return slow::sqrt(m_R2);
        }

    //! Get the wall boundary condition
    /*!
     * \returns Boundary condition at wall
     */
    HOSTDEVICE bool getNoSlip() const
        {
        return m_no_slip;
        }

#ifndef __HIPCC__
    //! Get the unique name of this geometry
    static std::string getName()
        {
        return std::string("Sphere");
        }
#endif // __HIPCC__

    private:
    const Scalar m_R2;    //!< Square of sphere radius
    const bool m_no_slip; //!< Boundary condition
    };

    } // namespace mpcd
    } // namespace hoomd

#undef HOSTDEVICE

#endif // MPCD_SPHERE_GEOMETRY_H_
