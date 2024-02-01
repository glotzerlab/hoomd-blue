// Copyright (c) 2009-2023 The Regents of the University of Michigan.
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
     * \param H Half-width between block regions.
     * \param w Half-width of blocks.
     */
    HOSTDEVICE BlockForce(Scalar F, Scalar H, Scalar w) : m_F(F)
        {
        m_H_plus_w = H + w;
        m_H_minus_w = H - w;
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

    //! Get the half separation distance between blocks
    Scalar getHalfSeparation() const
        {
        return 0.5 * (m_H_plus_w + m_H_minus_w);
        }

    //! Set the half separation distance between blocks
    void setHalfSeparation(Scalar H)
        {
        const Scalar w = getHalfWidth();
        m_H_plus_w = H + w;
        m_H_minus_w = H - w;
        }

    //! Get the block half width
    Scalar getHalfWidth() const
        {
        return 0.5 * (m_H_plus_w - m_H_minus_w);
        }

    //! Set the block half width
    void setHalfWidth(Scalar w)
        {
        const Scalar H = getHalfSeparation();
        m_H_plus_w = H + w;
        m_H_minus_w = H - w;
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
    Scalar m_H_plus_w;  //!< Upper bound on upper block, H + w
    Scalar m_H_minus_w; //!< Lower bound on upper block, H - w
    };

#ifndef __HIPCC__
namespace detail
    {
void export_BlockForce(pybind11::module& m);
    }  // end namespace detail
#endif // __HIPCC__

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE
#undef INLINE

#endif // MPCD_BLOCK_FORCE_H_
