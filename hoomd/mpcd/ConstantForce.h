// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ConstantForce.h
 * \brief Definition of mpcd::ConstantForce.
 */

#ifndef MPCD_CONSTANT_FORCE_H_
#define MPCD_CONSTANT_FORCE_H_

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
//! Constant force on all particles
class __attribute__((visibility("default"))) ConstantForce
    {
    public:
    //! Default constructor
    HOSTDEVICE ConstantForce() : m_F(make_scalar3(0, 0, 0)) { }

    //! Constructor
    /*!
     * \param F Force on all particles.
     */
    HOSTDEVICE ConstantForce(Scalar3 F) : m_F(F) { }

    //! Force evaluation method
    /*!
     * \param r Particle position.
     * \returns Force on the particle.
     *
     * Since the force is constant, just the constant value is returned.
     * More complicated functional forms will need to operate on \a r.
     */
    HOSTDEVICE INLINE Scalar3 evaluate(const Scalar3& r) const
        {
        return m_F;
        }

    HOSTDEVICE Scalar3 getForce() const
        {
        return m_F;
        }

    HOSTDEVICE void setForce(const Scalar3& F)
        {
        m_F = F;
        }

#ifndef __HIPCC__
    //! Get the unique name of this force
    static std::string getName()
        {
        return std::string("ConstantForce");
        }
#endif // __HIPCC__

    private:
    Scalar3 m_F; //!< Constant force
    };

#ifndef __HIPCC__
namespace detail
    {
void export_ConstantForce(pybind11::module& m);
    } // end namespace detail
#endif // __HIPCC__

    } // end namespace mpcd
    } // end namespace hoomd
#undef HOSTDEVICE
#undef INLINE

#endif // MPCD_CONSTANT_FORCE_H_
