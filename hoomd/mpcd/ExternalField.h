// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ExternalField.h
 * \brief Definition of mpcd::ExternalField.
 */

#ifndef MPCD_EXTERNAL_FIELD_H_
#define MPCD_EXTERNAL_FIELD_H_

#include "hoomd/HOOMDMath.h"

#ifdef NVCC
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"
#endif

namespace mpcd
{

//! External force field on MPCD particles.
/*!
 * The external field specifies a force that acts on the MPCD particles.
 * It will be evaluated inside the streaming kernel to accelerate particles.
 *
 * This is the abstract base class. Deriving classes must implement their
 * own evaluate() method, which takes a particle position and returns the force.
 *
 * You should explicitly instantiate device_new for your derived class in ExternalField.cu.
 * You should then add an appropriate overloaded ::reset() method to ExternalField.cc to
 * export the field to python. Then, add the python class to construct it. See ConstantForce
 * as an example.
 *
 * \warning
 * Because of the way NVCC handles compilation (see ExternalField.cu), new ExternalFields
 * can only be implemented within HOOMD and \b NOT through the plugin interface.
 */
class ExternalField
    {
    public:
        //! Virtual destructor (does nothing)
        HOSTDEVICE virtual ~ExternalField() {}

        //! Force evaluation method
        /*!
         * \param r Particle position.
         * \returns Force on the particle.
         *
         * Deriving classes must implement their own evaluate method.
         */
        HOSTDEVICE virtual Scalar3 evaluate(const Scalar3& r) const = 0;
    };

//! Constant, opposite force applied to particles in a block
/*!
 * Imposes a constant force in x as a function of position in z:
 *
 * \f{eqnarray*}
 *      \mathbf{F} &= +F \mathbf{e}_x & H-w \le z < H+w \\
 *                 &= -F \mathbf{e}_x & -H-w \le z < -H+w \\
 *                 &=    \mathbf{0}  & \mathrm{otherwise}
 * \f}
 *
 * where \a F is the force magnitude, \a H is the half-width between the
 * block centers, and \a w is the block half-width.
 *
 * This force field can be used to implement the double-parabola method for measuring
 * viscosity by setting \f$H = L_z/4\f$ and \f$w=L_z/4\f$, or to mimick the reverse
 * nonequilibrium shear flow profile by setting \f$H = L_z/4\f$ and \a w to a small value.
 */
class BlockForce : public ExternalField
    {
    public:
        //! Constructor
        /*!
         * \param F Force on all particles.
         * \param H Half-width between block regions.
         * \param w Half-width of blocks.
         */
        HOSTDEVICE BlockForce(Scalar F, Scalar H, Scalar w)
            : m_F(F)
            {
            m_H_plus_w = H+w;
            m_H_minus_w = H-w;
            }

        //! Force evaluation method
        /*!
         * \param r Particle position.
         * \returns Force on the particle.
         */
        HOSTDEVICE virtual Scalar3 evaluate(const Scalar3& r) const override
            {
            // sign = +1 if in top slab, -1 if in bottom slab, 0 if neither
            const signed char sign = (r.z >= m_H_minus_w && r.z < m_H_plus_w) - (r.z >= -m_H_plus_w && r.z < -m_H_minus_w);
            return make_scalar3(sign*m_F,0,0);
            }

    private:
        Scalar m_F;         //!< Constant force
        Scalar m_H_plus_w;  //!< Upper bound on upper block, H + w
        Scalar m_H_minus_w; //!< Lower bound on upper block, H - w
    };

//! Constant force on all particles
class ConstantForce : public ExternalField
    {
    public:
        //! Constructor
        /*!
         * \param F Force on all particles.
         */
        HOSTDEVICE ConstantForce(Scalar3 F) : m_F(F) {}

        //! Force evaluation method
        /*!
         * \param r Particle position.
         * \returns Force on the particle.
         *
         * Since the force is constant, just the constant value is returned.
         * More complicated functional forms will need to operate on \a r.
         */
        HOSTDEVICE virtual Scalar3 evaluate(const Scalar3& r) const override
            {
            return m_F;
            }

    private:
        Scalar3 m_F;    //!< Constant force
    };

//! Shearing sine force
/*!
 * Imposes a sinusoidally varying force in x as a function of position in z.
 * The shape of the force is controlled by the amplitude and the wavenumber.
 *
 * \f[
 * \mathbf{F}(\mathbf{r}) = F \sin (k r_z) \mathbf{e}_x
 * \f]
 */
class SineForce : public ExternalField
    {
    public:
        //! Constructor
        /*!
         * \param F Amplitude of the force.
         * \param k Wavenumber for the force.
         */
        HOSTDEVICE SineForce(Scalar F, Scalar k)
            : m_F(F), m_k(k)
            {}

            //! Force evaluation method
            /*!
             * \param r Particle position.
             * \returns Force on the particle.
             *
             * Specifies the force to act in x as a function of z. Fast math
             * routines are used since this is probably sufficiently accurate,
             * given the other numerical errors already present.
             */
        HOSTDEVICE virtual Scalar3 evaluate(const Scalar3& r) const override
            {
            return make_scalar3(m_F*fast::sin(m_k*r.z),0,0);
            }

    private:
        Scalar m_F; //!< Force constant
        Scalar m_k; //!< Wavenumber for force in z
    };

#ifndef NVCC
namespace detail
{
void export_ExternalFieldPolymorph(pybind11::module& m);
} // end namespace detail
#endif // NVCC

} // end namespace mpcd

#undef HOSTDEVICE

#endif // MPCD_EXTERNAL_FIELD_H_
