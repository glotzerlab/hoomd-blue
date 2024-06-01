// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SineForce.h
 * \brief Definition of mpcd::SineForce.
 */

#ifndef MPCD_SINE_FORCE_H_
#define MPCD_SINE_FORCE_H_

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

//! Shearing sine force
/*!
 * Imposes a sinusoidally varying force in x as a function of position in y.
 * The shape of the force is controlled by the amplitude and the wavenumber.
 *
 * \f[
 * \mathbf{F}(\mathbf{r}) = F \sin (k r_y) \mathbf{e}_x
 * \f]
 */
class __attribute__((visibility("default"))) SineForce
    {
    public:
    //! Default constructor
    HOSTDEVICE SineForce() : SineForce(0, 0) { }

    //! Constructor
    /*!
     * \param F Amplitude of the force.
     * \param k Wavenumber for the force.
     */
    HOSTDEVICE SineForce(Scalar F, Scalar k) : m_F(F), m_k(k) { }

    //! Force evaluation method
    /*!
     * \param r Particle position.
     * \returns Force on the particle.
     *
     * Specifies the force to act in x as a function of y. Fast math
     * routines are used since this is probably sufficiently accurate,
     * given the other numerical errors already present.
     */
    HOSTDEVICE Scalar3 evaluate(const Scalar3& r) const
        {
        return make_scalar3(m_F * fast::sin(m_k * r.y), 0, 0);
        }

    //! Get the sine amplitude
    Scalar getAmplitude() const
        {
        return m_F;
        }

    //! Set the sine amplitude
    void setAmplitude(Scalar F)
        {
        m_F = F;
        }

    //! Get the sine wavenumber
    Scalar getWavenumber() const
        {
        return m_k;
        }

    //! Set the sine wavenumber
    void setWavenumber(Scalar k)
        {
        m_k = k;
        }

#ifndef __HIPCC__
    //! Get the unique name of this force
    static std::string getName()
        {
        return std::string("SineForce");
        }
#endif // __HIPCC__

    private:
    Scalar m_F; //!< Force constant
    Scalar m_k; //!< Wavenumber for force in y
    };

#ifndef __HIPCC__
namespace detail
    {
void export_SineForce(pybind11::module& m);
    } // end namespace detail
#endif // __HIPCC__

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE
#undef INLINE

#endif // MPCD_SINE_FORCE_H_
