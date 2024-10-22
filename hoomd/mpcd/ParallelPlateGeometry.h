// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParallelPlateGeometry.h
 * \brief Definition of the MPCD slit channel geometry
 */

#ifndef MPCD_PARALLEL_PLATE_GEOMETRY_H_
#define MPCD_PARALLEL_PLATE_GEOMETRY_H_

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
//! Parallel plate (slit) geometry
/*!
 * This class defines the geometry consistent with two infinite parallel plates. When the plates are
 * in relative motion, Couette flow can be generated in the channel. If a uniform body force is
 * applied to the fluid, the parabolic Poiseuille flow profile is created. Both flow profiles
 * require the enforcement of no-slip boundary conditions.
 *
 * The channel geometry is defined by two parameters: the channel half-width \a H, and the velocity
 * of the plates \a V. The total distance between the plates is \f$2H\f$. The plates are stacked in
 * the \a y direction, and are centered about the origin \f$y=0\f$. The upper plate moves in the
 * \f$+x\f$ direction with velocity \a V, and the lower plate moves in the \f$-x\f$ direction with
 * velocity \f$-V\f$. Hence, for no-slip boundary conditions there is a velocity profile:
 *
 * \f[
 *      v_x(y) = \frac{Vy}{H}
 * \f]
 *
 * This gives an effective shear rate \f$\dot\gamma = V/H\f$, and the shear stress is
 * $\sigma_{xy}\f$.
 *
 * The geometry enforces boundary conditions \b only on the MPCD solvent particles. Additional
 * interactions are required with any embedded particles using appropriate wall potentials.
 *
 * The wall boundary conditions can optionally be changed to slip conditions. For these BCs, the
 * previous discussion of the various flow profiles no longer applies.
 */
class __attribute__((visibility("default"))) ParallelPlateGeometry
    {
    public:
    //! Constructor
    /*!
     * \param separation Distance between plates
     * \param speed Speed of the wall
     * \param no_slip Boundary condition at the wall (slip or no-slip)
     */
    HOSTDEVICE ParallelPlateGeometry(Scalar separation, Scalar speed, bool no_slip)
        : m_H(Scalar(0.5) * separation), m_V(speed), m_no_slip(no_slip)
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
        /*
         * Detect if particle has left the box, and try to avoid branching or absolute value calls.
         * The sign used in calculations is +1 if the particle is out-of-bounds in the +y direction,
         * -1 if the particle is out-of-bounds in the -y direction, and 0 otherwise.
         *
         * We intentionally use > / < rather than >= / <= to make sure that spurious collisions do
         * not get detected when a particle is reset to the boundary location. A particle landing
         * exactly on the boundary from the bulk can be immediately reflected on the next streaming
         * step, and so the motion is essentially equivalent up to an epsilon of difference in the
         * channel width.
         */
        const signed char sign = (char)((pos.y > m_H) - (pos.y < -m_H));
        // exit immediately if no collision is found or particle is not moving normal to the wall
        // (since no new collision could have occurred if there is no normal motion)
        if (sign == 0 || vel.y == Scalar(0))
            {
            dt = Scalar(0);
            return false;
            }

        /*
         * Remaining integration time dt is amount of time spent traveling distance out of bounds.
         * If sign = +1, then pos.y > H. If sign = -1, then pos.y < -H, and we need difference in
         * the opposite direction.
         *
         * TODO: if dt < 0, it is a spurious collision. How should it be treated?
         */
        dt = (pos.y - sign * m_H) / vel.y;

        // backtrack the particle for dt to get to point of contact
        pos.x -= vel.x * dt;
        pos.y = sign * m_H;
        pos.z -= vel.z * dt;

        // update velocity according to boundary conditions
        // no-slip requires reflection of the tangential components
        if (m_no_slip)
            {
            vel.x = -vel.x + Scalar(sign * 2) * m_V;
            vel.z = -vel.z;
            }
        // both slip and no-slip have no penetration of the surface
        vel.y = -vel.y;

        return true;
        }

    //! Check if a particle is out of bounds
    /*!
     * \param pos Current particle position
     * \returns True if particle is out of bounds, and false otherwise
     */
    HOSTDEVICE bool isOutside(const Scalar3& pos) const
        {
        return (pos.y > m_H || pos.y < -m_H);
        }

    //! Add a contribution to random virtual particle velocity.
    /*!
     * \param vel Velocity of virtual particle
     * \param pos Position of virtual particle
     *
     * Add velocity of moving parallel plates to \a vel for particles beyond the separation
     * distance. The velocity is increased by \a m_V if beyond the upper plate, and decreased by \a
     * m_V if beyond the lower plate.
     */
    HOSTDEVICE void addToVirtualParticleVelocity(Scalar3& vel, const Scalar3& pos) const
        {
        if (pos.y > m_H)
            {
            vel.x += m_V;
            }
        else if (pos.y < -m_H)
            {
            vel.x -= m_V;
            }
        }

    //! Get channel half width
    /*!
     * \returns Channel half width
     */
    HOSTDEVICE Scalar getH() const
        {
        return m_H;
        }

    //! Get distance between plates
    /*!
     * \returns Distance between plates
     */
    HOSTDEVICE Scalar getSeparation() const
        {
        return 2 * m_H;
        }

    //! Get the wall speed
    /*!
     * \returns Wall speed
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
        return std::string("ParallelPlates");
        }
#endif // __HIPCC__

    private:
    const Scalar m_H;     //!< Half of the channel width
    const Scalar m_V;     //!< Velocity of the wall
    const bool m_no_slip; //!< Boundary condition
    };

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE

#endif // MPCD_PARALLEL_PLATE_GEOMETRY_H_
