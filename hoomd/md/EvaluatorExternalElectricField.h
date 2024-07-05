// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_EXTERNAL_ELECTRIC_FIELD_H__
#define __EVALUATOR_EXTERNAL_ELECTRIC_FIELD_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <math.h>

/*! \file EvaluatorExternalElectricField.h
    \brief Defines the external potential evaluator to induce a periodic ordered phase
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
//! Class for evaluating an electric field
/*! <b>General Overview</b>
    The external potential \f$V(\vec{r}) \f$ is implemented using the following formula:

    \f[
    V(\vec{r}) = - q_i \vec{E} \cdot \vec{r}
    \f]

    where \f$E\f$ is the strength of the electric field and \f$q_i\f$ is the charge of particle i.
*/
class EvaluatorExternalElectricField
    {
    public:
    //! type of parameters this external potential accepts
    struct param_type
        {
        Scalar3 E;

#ifndef __HIPCC__
        param_type() : E(make_scalar3(0, 0, 0)) { }

        param_type(pybind11::object params)
            {
            pybind11::tuple py_E(params);
            E.x = pybind11::cast<Scalar>(py_E[0]);
            E.y = pybind11::cast<Scalar>(py_E[1]);
            E.z = pybind11::cast<Scalar>(py_E[2]);
            }

        pybind11::object toPython()
            {
            pybind11::tuple params;
            params = pybind11::make_tuple(E.x, E.y, E.z);
            return std::move(params);
            }
#endif // ifndef __HIPCC__
        } __attribute__((aligned(16)));

    typedef void* field_type;

    //! Constructs the constraint evaluator
    /*! \param X position of particle
        \param box box dimensions
        \param params per-type parameters of external potential
    */
    DEVICE EvaluatorExternalElectricField(Scalar3 X,
                                          quat<Scalar> q,
                                          const BoxDim& box,
                                          const param_type& params,
                                          const field_type& field)
        : m_pos(X), m_box(box), m_E(params.E)
        {
        }

    DEVICE static bool isAnisotropic()
        {
        return false;
        }

    //! ExternalElectricField needs charges
    DEVICE static bool needsCharge()
        {
        return true;
        }

    //! Accept the optional charge value
    /*! \param qi Charge of particle i
     */
    DEVICE void setCharge(Scalar qi)
        {
        m_qi = qi;
        }

    //! Declares additional virial contributions are needed for the external field
    /*! No contribution
     */
    DEVICE static bool requestFieldVirialTerm()
        {
        return true;
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
        F = m_qi * m_E;
        energy = -m_qi * dot(m_E, m_pos);

        virial[0] = F.x * m_pos.x;
        virial[1] = F.x * m_pos.y;
        virial[2] = F.x * m_pos.z;
        virial[3] = F.y * m_pos.y;
        virial[4] = F.y * m_pos.z;
        virial[5] = F.z * m_pos.z;

        T.x = Scalar(0.0);
        T.y = Scalar(0.0);
        T.z = Scalar(0.0);
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("e_field");
        }
#endif

    protected:
    Scalar3 m_pos; //!< particle position
    BoxDim m_box;  //!< box dimensions
    Scalar m_qi;   //!< particle charge
    Scalar3 m_E;   //!< the field vector
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_EXTERNAL_LAMELLAR_H__
