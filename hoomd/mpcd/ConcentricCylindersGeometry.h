// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ConcentricCylindersGeometry.h
 * \brief Definition of the MPCD slit channel geometry
 */

#ifndef MPCD_CONCENTRIC_CYLINDERS_GEOMETRY_H_
#define MPCD_CONCENTRIC_CYLINDERS_GEOMETRY_H_

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
//! Concentric cylinders geometry

class __attribute__((visibility("default"))) ConcentricCylindersGeometry
    {
    public:
    //! Constructor
    /*!
     * \param R0 Inner radius of the cylinder
     * \param R1 Outer radius of the cylinder
     * \param angular_speed Angular speed of the outer cylinder
     * \param no_slip Boundary condition at the wall (slip or no-slip)
     */
    HOSTDEVICE ConcentricCylindersGeometry(Scalar R0, Scalar R1, Scalar angular_speed, bool no_slip)
        : m_R0_sq(R0 * R0), m_R1_sq(R1 * R1), m_w(angular_speed), m_no_slip(no_slip)
        {
        }

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
        const Scalar rsq = pos.x * pos.x + pos.y * pos.y;
        /*
         * Check if particle is in bounds
         */
        const signed char sign = (char)((rsq < m_R0_sq) - (rsq > m_R1_sq));
        // exit immediately if no collision is found or particle is stationary in xy plane
        const Scalar vxy_sq = vel.x * vel.x + vel.y * vel.y;
        if (sign == 0 || vxy_sq == Scalar(0))
            {
            dt = Scalar(0);
            return false;
            }

        /*
         * Find the time remaining when the particle is collided with wall. This time is computed
         * by backtracking the position, dot(pos, pos) = R^2.
         */
        const Scalar Rsq = (sign == 1) ? m_R0_sq : m_R1_sq;
        const Scalar b = -Scalar(2) * (pos.x * vel.x + pos.y * vel.y);
        const Scalar c = pos.x * pos.x + pos.y * pos.y - Rsq;
        dt = (-b + sign * (slow::sqrt(b * b - Scalar(4) * vxy_sq * c))) / (Scalar(2) * vxy_sq);

        // backtrack the particle for dt to get to point of contact
        pos -= vel * dt;

        // update velocity according to boundary conditions
        // no-slip requires reflection of the tangential components.
        if (m_no_slip)
            {
            vel = -vel;
            // if the particle hits the moving outside wall, then vel = -vel + 2V_t.
            /*
             * t = [-y/R, x/R, 0] is the tangential unit vector at the point of contact r, which is
             * the particle position that has been backtracked to the surface in the previous step.
             * Therefore, V_t = m_w * R * t = m_w * [-pos.y, pos.x, 0].
             */
            if (sign == -1)
                {
                vel.x += Scalar(2) * m_w * -pos.y;
                vel.y += Scalar(2) * m_w * pos.x;
                }
            }
        // slip requires reflection of the normal components.
        /*
         * n = r/R is the normal unit vector at the point of contact r, which is the particle
         * position. Therefore, V_n = dot(v, n) * n = vel * pos * pos / Rsq.
         */
        else
            {
            vel.x -= Scalar(2) * pos.x * pos.x * vel.x / Rsq;
            vel.y -= Scalar(2) * pos.y * pos.y * vel.y / Rsq;
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
        const Scalar rsq = pos.x * pos.x + pos.y * pos.y;
        return (rsq > m_R1_sq || rsq < m_R0_sq);
        }

    //! Get inner radius of the cylinder
    /*!
     * \returns Inner radius of the cylinder
     */
    HOSTDEVICE Scalar getInnerRadius() const
        {
        return slow::sqrt(m_R0_sq);
        }

    //! Get outer radius of the cylinder
    /*!
     * \returns Outer radius of the cylinder
     */
    HOSTDEVICE Scalar getOuterRadius() const
        {
        return slow::sqrt(m_R1_sq);
        }

    //! Get the speed of the outer cylinder
    /*!
     * \returns Speed of the outer cylinder
     */
    HOSTDEVICE Scalar getAngularSpeed() const
        {
        return m_w;
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
        return std::string("ConcentricCylinders");
        }
#endif // __HIPCC__

    private:
    const Scalar m_R0_sq; //!< Square of inner radius
    const Scalar m_R1_sq; //!< Square of outer radius
    const Scalar m_w;     //!< Angular velocity of the outer wall
    const bool m_no_slip; //!< Boundary condition
    };

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE

#endif // MPCD_CONCENTRIC_CYLINDERS_H_
