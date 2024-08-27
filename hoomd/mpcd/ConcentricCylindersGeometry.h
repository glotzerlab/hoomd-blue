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
     * \param speed Speed of the outer cylinder
     * \param no_slip Boundary condition at the wall (slip or no-slip)
     */
    HOSTDEVICE ConcentricCylindersGeometry(Scalar R0, Scalar R1, Scalar speed, bool no_slip)
        : m_R0(R0), m_R02(R0 * R0), m_R1(R1), m_R12(R1 * R1), m_V(speed), m_no_slip(no_slip)
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
        const Scalar r = pos.x * pos.x + pos.y * pos.y;
        /*
         * Check if particle is in bounds
         */
        const signed char sign = (char)((r > m_R12) - (r < m_R02));
        // exit immediately if no collision is found or particle is not moving normal to the wall
        // (since no new collision could have occurred if there is no normal motion)
        if (sign == 0 || vel.x * vel.x + vel.y * vel.y == Scalar(0))
            {
            dt = Scalar(0);
            return false;
            }

        /*
         * Find the time remaining when the particle is collided with wall. This time is computed
         * by backtracking the position, dot(pos, pos) = R^2.
         */
        const Scalar R = (sign == 1) ? R12 : R02;
        const Scalar a = vel.x * vel.x + vel.y * vel.y;
        const Scalar b = -Scalar(2) * (pos.x * vel.x + pos.y * vel.y);
        const Scalar c = pos.x * pos.x + pos.y * pos.y - R * R;
        dt = (-b + slow::sqrt(b * b - Scalar(4) * a * c)) / (Scalar(2) * a);

        // backtrack the particle for dt to get to point of contact
        pos -= vel * dt;

        // update velocity according to boundary conditions
        // no-slip requires reflection of the tangential components.
        if (sign == 1)
            {
            // no-slip requires reflection of the tangential components.
            if (m_no_slip)
                {
                vel.x = -vel.x + Scalar(2) * m_V * pos.x / R1;
                vel.y = -vel.y + Scalar(2) * m_V * pos.y / R1;
                vel.z = -vel.z;
                }
            // slip requires reflection of the normal components.
            else
                {
                vel.x -= Scalar(2) * pos.x * pos.y * vel.x / (R * R);
                vel.y -= Scalar(2) * pos.y * pos.y * vel.y / (R * R);
                }
            }

        if (sign == -1)
            {
            if (m_no_slip)
                {
                vel = -vel;
                }
            else
                {
                vel.x -= Scalar(2) * pos.x * pos.y * vel.x / (R * R);
                vel.y -= Scalar(2) * pos.y * pos.y * vel.y / (R * R);
                }
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
        const Scalar r = pos.x * pos.x + pos.y * pos.y return (r > m_R12 || r < m_R02);
        }

    //! Get inner radius of the cylinder
    /*!
     * \returns Inner radius of the cylinder
     */
    HOSTDEVICE Scalar getR0() const
        {
        return m_R0;
        }

    //! Get outer radius of the cylinder
    /*!
     * \returns Outer radius of the cylinder
     */
    HOSTDEVICE Scalar getR1() const
        {
        return m_R1;
        }

    //! Get the speed of the outer cylinder
    /*!
     * \returns Speed of the outer cylinder
     */
    HOSTDEVICE Scalar getSpeed() const
        {
        return m_V;
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
    const Scalar m_R0;    //!< Inner radius of the cylinder
    const Scalar m_R02;   //!< Square of inner radius
    const Scalar m_R1;    //!< Outer radius of the cylinder
    const Scalar m_R12;   //!< Velocity of the wall
    const bool m_no_slip; //!< Boundary condition
    };

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE

#endif // MPCD_CONCENTRIC_CYLINDERS_H_
