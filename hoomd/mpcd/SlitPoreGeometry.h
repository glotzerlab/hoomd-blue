// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SlitPoreGeometry.h
 * \brief Definition of the MPCD slit pore geometry
 */

#ifndef MPCD_SLIT_PORE_GEOMETRY_H_
#define MPCD_SLIT_PORE_GEOMETRY_H_

#include "BoundaryCondition.h"

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
namespace detail
    {
//! Parallel plate (slit) geometry with pore boundaries
/*!
 * This class defines the geometry consistent with two finite-length parallel plates. The plates
 * are finite in \a x, infinite in \a y, and stacked in \a z. If a uniform body force is applied
 * to the fluid and there are no-slip boundary conditions, parabolic Poiseuille flow profile is
 * created subject to entrance/exit effects.
 *
 * The channel geometry is defined by two parameters: the channel half-width \a H in \a z and the
 * pore half-length \a L in \a x. The total distance between the plates is \f$2H\f$, and their total
 * length is \f$2L\f$. The plates are centered about the origin \f$(x,z)=(0,0)\f$.
 *
 * There is an infinite bounding wall with normal in \a x at the edges of the pore.
 * There are no bounding walls away from the pore (PBCs apply).
 *
 * \sa mpcd::detail::SlitGeometry for additional discussion of the boundary conditions, etc.
 */
class __attribute__((visibility("default"))) SlitPoreGeometry
    {
    public:
    //! Constructor
    /*!
     * \param H Channel half-width
     * \param L Pore half-length
     * \param bc Boundary condition at the wall (slip or no-slip)
     */
    HOSTDEVICE SlitPoreGeometry(Scalar H, Scalar L, boundary bc) : m_H(H), m_L(L), m_bc(bc) { }

    //! Detect collision between the particle and the boundary
    /*!
     * \param pos Proposed particle position
     * \param vel Proposed particle velocity
     * \param dt Integration time remaining (inout).
     *
     * \returns True if a collision occurred, and false otherwise
     *
     * \post The particle position \a pos is moved to the point of reflection, the velocity \a vel
     * is updated according to the appropriate bounce back rule, and the integration time \a dt is
     * decreased to the amount of time remaining.
     *
     * The passed value of \a dt must be the time taken to arrive at pos. The returned value of \a
     * dt will be less than this time.
     */
    HOSTDEVICE bool detectCollision(Scalar3& pos, Scalar3& vel, Scalar& dt) const
        {
        /* First check that the particle ended up inside the pore or walls.
         * sign.x is +1 if outside pore in +x, -1 if outside pore in -x, and 0 otherwise.
         * sign.y is +1 if outside walls in +z, -1 if outside walls in -z, and 0 otherwise. */
        const char2 sign = make_char2((char)((pos.x >= m_L) - (pos.x <= -m_L)),
                                      (char)((pos.z > m_H) - (pos.z < -m_H)));
        // exit early if collision didn't happen
        if (sign.x != 0 || sign.y == 0)
            {
            dt = Scalar(0);
            return false;
            }

        // times to either hit the pore in x, or the wall in z
        Scalar2 dts;
        if (vel.x != Scalar(0))
            {
            // based on direction moving, could only have hit one of these edges
            const Scalar xw = (vel.x > 0) ? -m_L : m_L;
            dts.x = (pos.x - xw) / vel.x;
            }
        else
            {
            // no collision
            dts.x = Scalar(-1);
            }
        if (vel.z != Scalar(0))
            {
            dts.y = (pos.z - sign.y * m_H) / vel.z;
            }
        else
            {
            // no collision
            dts.y = Scalar(-1);
            }

        // determine if collisions happend with the x or z walls.
        // if both occur, use the one that happened first (leaves the most time)
        // this neglects coming through the corner exactly, but that's OK to an approx.
        uchar2 hits = make_uchar2(dts.x > 0 && dts.x < dt, dts.y > 0 && dts.y < dt);
        if (hits.x && hits.y)
            {
            if (dts.x < dts.y)
                {
                hits.x = 0;
                }
            else
                {
                hits.y = 0;
                }
            }

        // set integration time and normal based on the collision
        Scalar3 n = make_scalar3(0, 0, 0);
        if (hits.x && !hits.y)
            {
            dt = dts.x;
            // 1 if v.x < 0 (right pore wall), -1 if v.x > 0 (left pore wall)
            n.x = (vel.x < 0) - (vel.x > 0);
            }
        else if (!hits.x && hits.y)
            {
            dt = dts.y;
            n.z = -sign.y;
            }
        else
            {
            dt = Scalar(0);
            return false;
            }

        // backtrack the particle for dt to get to point of contact
        pos -= vel * dt;

        // update velocity according to boundary conditions
        // no-slip requires reflection of the tangential components
        const Scalar3 vn = dot(n, vel) * n;
        if (m_bc == boundary::no_slip)
            {
            const Scalar3 vt = vel - vn;
            vel += Scalar(-2) * vt;
            }
        // always reflect normal component for no-penetration
        vel += Scalar(-2) * vn;

        return true;
        }

    //! Check if a particle is out of bounds
    /*!
     * \param pos Current particle position
     * \returns True if particle is out of bounds, and false otherwise
     */
    HOSTDEVICE bool isOutside(const Scalar3& pos) const
        {
        return ((pos.x > -m_L && pos.x < m_L) && (pos.z > m_H || pos.z < -m_H));
        }

    //! Validate that the simulation box is large enough for the geometry
    /*!
     * \param box Global simulation box
     * \param cell_size Size of MPCD cell
     *
     * The box is large enough for the pore if it is padded along the x and z direction so that
     * the cells just outside the pore would not interact with each other through the boundary.
     */
    HOSTDEVICE bool validateBox(const BoxDim& box, Scalar cell_size) const
        {
        const Scalar3 hi = box.getHi();
        const Scalar3 lo = box.getLo();

        return ((hi.x - m_L) >= cell_size && (-m_L - lo.x) >= cell_size && (hi.z - m_H) >= cell_size
                && (-m_H - lo.z) >= cell_size);
        }

    //! Get pore half width
    /*!
     * \returns Pore half width
     */
    HOSTDEVICE Scalar getH() const
        {
        return m_H;
        }

    //! Get pore half length
    /*!
     * \returns Pore half length
     */
    HOSTDEVICE Scalar getL() const
        {
        return m_L;
        }

    //! Get the wall boundary condition
    /*!
     * \returns Boundary condition at wall
     */
    HOSTDEVICE boundary getBoundaryCondition() const
        {
        return m_bc;
        }

#ifndef __HIPCC__
    //! Get the unique name of this geometry
    static std::string getName()
        {
        return std::string("SlitPore");
        }
#endif // __HIPCC__

    private:
    const Scalar m_H;    //!< Half of the channel width
    const Scalar m_L;    //!< Half of the pore length
    const boundary m_bc; //!< Boundary condition
    };

    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE

#endif // MPCD_SLIT_PORE_GEOMETRY_H_
