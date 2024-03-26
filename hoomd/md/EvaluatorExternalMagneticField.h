// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_EXTERNAL_MAGNETIC_FIELD_H__
#define __EVALUATOR_EXTERNAL_MAGNETIC_FIELD_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <math.h>

/*! \file EvaluatorExternalMagneticField.h
    \brief Defines the external potential evaluator to induce a magnetic field
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
//! Class for evaluating an magnetic field
/*! <b>General Overview</b>
    The external potential \f$V(\theta) \f$ is implemented using the following formula:

    \f[
    V(\theta}) = - \vec{B} \cdot \vec{n}_i(\theta)
    \f]

    where \f$B\f$ is the strength of the magnetic field and \f$\vec{n}_i\f$ is the magnetic moment
   of particle i.
*/
class EvaluatorExternalMagneticField
    {
    public:
    //! type of parameters this external potential accepts
    struct param_type
        {
        Scalar3 B;
        Scalar3 mu;

#ifndef __HIPCC__
        param_type() : B(make_scalar3(0, 0, 0)), mu(make_scalar3(0, 0, 0)) { }

        param_type(pybind11::dict params)
            {
            pybind11::tuple py_B = params["B"];
            B.x = pybind11::cast<Scalar>(py_B[0]);
            B.y = pybind11::cast<Scalar>(py_B[1]);
            B.z = pybind11::cast<Scalar>(py_B[2]);
            pybind11::tuple py_mu = params["mu"];
            mu.x = pybind11::cast<Scalar>(py_mu[0]);
            mu.y = pybind11::cast<Scalar>(py_mu[1]);
            mu.z = pybind11::cast<Scalar>(py_mu[2]);
            }

        pybind11::dict toPython()
            {
            pybind11::dict d;
            d["B"] = pybind11::make_tuple(B.x, B.y, B.z);
            d["mu"] = pybind11::make_tuple(mu.x, mu.y, mu.z);
            return d;
            }
#endif // ifndef __HIPCC__
        } __attribute__((aligned(16)));

    typedef void* field_type;

    //! Constructs the external field evaluator
    /*! \param X position of particle
        \param box box dimensions
        \param params per-type parameters of external potential
    */
    DEVICE EvaluatorExternalMagneticField(Scalar3 X,
                                          quat<Scalar> q,
                                          const BoxDim& box,
                                          const param_type& params,
                                          const field_type& field)
        : m_q(q), m_B(params.B), m_mu(params.mu)
        {
        }

    DEVICE static bool isAnisotropic()
        {
        return true;
        }

    //! ExternalMagneticField needs charges
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge value
    /*! \param qi Charge of particle i
     */
    DEVICE void setCharge(Scalar qi) { }

    //! Declares additional virial contributions are needed for the external field
    /*! No contribution
     */
    DEVICE static bool requestFieldVirialTerm()
        {
        return false;
        }

    //! Evaluate the force, energy and virial
    /*! \param F force vector
        \param T torque vector
        \param energy value of the energy
        \param virial array of six scalars for the upper triangular virial tensor
    */
    DEVICE void
    evalForceTorqueEnergyAndVirial(Scalar3& F, Scalar3& T, Scalar& energy, Scalar* virial)
        {
        vec3<Scalar> dir = rotate(m_q, m_mu);

        vec3<Scalar> T_vec = cross(dir, m_B);

        T.x = T_vec.x;
        T.y = T_vec.y;
        T.z = T_vec.z;

        energy = -dot(dir, m_B);

        F.x = Scalar(0.0);
        F.y = Scalar(0.0);
        F.z = Scalar(0.0);

        for (unsigned int i = 0; i < 6; i++)
            virial[i] = Scalar(0.0);
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("b_field");
        }
#endif

    protected:
    quat<Scalar> m_q;  //!< Particle orientation
    vec3<Scalar> m_B;  //!< Magnetic field vector (box frame).
    vec3<Scalar> m_mu; //!< Magnetic dipole moment (particle frame).
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_EXTERNAL_LAMELLAR_H__
