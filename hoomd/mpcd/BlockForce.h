// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/BlockForce.h
 * \brief Definition of mpcd::BlockForce.
 */

#ifndef MPCD_BLOCK_FORCE_H_
#define MPCD_BLOCK_FORCE_H_

#include "hoomd/HOOMDMath.h"

#ifdef __HIPCC__
#define HOSTDEVICE __host__ __device__
#define INLINE inline
#else
#define HOSTDEVICE
#define INLINE inline __attribute__((always_inline))
#include <pybind11/pybind11.h>
#include <string>
#endif

namespace hoomd
    {
namespace mpcd
    {

//! Constant, opposite force applied to particles in a block
/*!
 * Imposes a constant force in x as a function of position in y:
 *
 * \f{eqnarray*}
 *      \mathbf{F} &= +F \mathbf{e}_x & H-w \le y < H+w \\
 *                 &= -F \mathbf{e}_x & -H-w \le y < -H+w \\
 *                 &=    \mathbf{0}  & \mathrm{otherwise}
 * \f}
 *
 * where \a F is the force magnitude, \a H is the half-width between the
 * block centers, and \a w is the block half-width.
 *
 * This force field can be used to implement the double-parabola method for measuring
 * viscosity by setting \f$H = L_y/4\f$ and \f$w=L_y/4\f$, or to mimick the reverse
 * nonequilibrium shear flow profile by setting \f$H = L_y/4\f$ and \a w to a small value.
 */
class __attribute__((visibility("default"))) BlockForce
    {
    public:
    //! Default constructor
    HOSTDEVICE BlockForce() : BlockForce(0, 0, 0) { }

    //! Constructor
    /*!
     * \param F Force on all particles.
     * \param separation Separation between centers of blocks.
     * \param width Width of each block.
     */
    HOSTDEVICE BlockForce(Scalar F, Scalar separation, Scalar width) : m_F(F)
        {
        m_H_plus_w = Scalar(0.5) * (separation + width);
        m_H_minus_w = Scalar(0.5) * (separation - width);
        }

    //! Force evaluation method
    /*!
     * \param r Particle position.
     * \returns Force on the particle.
     */
    HOSTDEVICE Scalar3 evaluate(const Scalar3& r) const
        {
        // sign = +1 if in top slab, -1 if in bottom slab, 0 if neither
        const signed char sign = (char)((r.y >= m_H_minus_w && r.y < m_H_plus_w)
                                        - (r.y >= -m_H_plus_w && r.y < -m_H_minus_w));
        return make_scalar3(sign * m_F, 0, 0);
        }

    //! Get the force in the block
    Scalar getForce() const
        {
        return m_F;
        }

    //! Set the force in the block
    void setForce(Scalar F)
        {
        m_F = F;
        }

    //! Get the separation distance between block centers
    Scalar getSeparation() const
        {
        return (m_H_plus_w + m_H_minus_w);
        }

    //! Set the separation distance between block centers
    void setSeparation(Scalar H)
        {
        const Scalar w = getWidth();
        m_H_plus_w = Scalar(0.5) * (H + w);
        m_H_minus_w = Scalar(0.5) * (H - w);
        }

    //! Get the block width
    Scalar getWidth() const
        {
        return (m_H_plus_w - m_H_minus_w);
        }

    //! Set the block width
    void setWidth(Scalar w)
        {
        const Scalar H = getSeparation();
        m_H_plus_w = Scalar(0.5) * (H + w);
        m_H_minus_w = Scalar(0.5) * (H - w);
        }

#ifndef __HIPCC__
    //! Get the unique name of this force
    static std::string getName()
        {
        return std::string("BlockForce");
        }
#endif // __HIPCC__

    private:
    Scalar m_F;         //!< Constant force
    Scalar m_H_plus_w;  //!< Upper bound on upper block
    Scalar m_H_minus_w; //!< Lower bound on upper block
    };
    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE
#undef INLINE

#endif // MPCD_BLOCK_FORCE_H_
