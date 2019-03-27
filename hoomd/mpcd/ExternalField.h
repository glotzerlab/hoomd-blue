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
 * There is currently a strange behavior with the external fields in CUDA code. In addition
 * to templating device_new here, you should also template it in any classes making use of
 * the ExternalField by adding it to the TEMPLATE_DEVICE_NEW_FIELDS macro. This may be revised in future.
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
 * \begin{equation}
 * \mathbf{F}(\mathbf{r}) = F \sin (k r_z) \mathbf{e}_x
 * \end{equation}
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

#ifdef NVCC
#define TEMPLATE_DEVICE_NEW_FIELDS \
template mpcd::ConstantForce* hoomd::gpu::device_new(Scalar3); \
template mpcd::SineForce* hoomd::gpu::device_new(Scalar,Scalar);
#endif // NVCC

#ifndef NVCC
namespace detail
{
void export_ExternalFieldPolymorph(pybind11::module& m);
} // end namespace detail
#endif // NVCC

} // end namespace mpcd

#undef HOSTDEVICE

#endif // MPCD_EXTERNAL_FIELD_H_
