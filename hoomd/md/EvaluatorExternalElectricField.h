// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __EVALUATOR_EXTERNAL_ELECTRIC_FIELD_H__
#define __EVALUATOR_EXTERNAL_ELECTRIC_FIELD_H__

#ifndef NVCC
#include <string>
#endif

#include <math.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

/*! \file EvaluatorExternalElectricField.h
    \brief Defines the external potential evaluator to induce a periodic ordered phase
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

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
        typedef struct param{} param_type;
        typedef Scalar3 field_type;

        //! Constructs the constraint evaluator
        /*! \param X position of particle
            \param box box dimensions
            \param params per-type parameters of external potential
        */
        DEVICE EvaluatorExternalElectricField(Scalar3 X, const BoxDim& box, const param_type& params, const field_type& field)
            : m_pos(X),
              m_box(box),
              m_field(field)
            {
            }

        //! External Periodic doesn't need diameters
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter value
        /*! \param di Diameter of particle i
        */
        DEVICE void setDiameter(Scalar di) { }

        //! External Periodic doesn't need charges
        DEVICE static bool needsCharge() { return true; }
        //! Accept the optional diameter value
        /*! \param qi Charge of particle i
        */
        DEVICE void setCharge(Scalar qi) { m_qi = qi; }

        //! Declares additional virial contributions are needed for the external field
        /*! No contribution
        */
        DEVICE static bool requestFieldVirialTerm() { return true; }

        //! Evaluate the force, energy and virial
        /*! \param F force vector
            \param energy value of the energy
            \param virial array of six scalars for the upper triangular virial tensor
        */
        DEVICE void evalForceEnergyAndVirial(Scalar3& F, Scalar& energy, Scalar* virial)
            {
            F = m_qi * m_field;
            energy = -m_qi * dot(m_field,m_pos);

            virial[0] = F.x*m_pos.x;
            virial[1] = F.x*m_pos.y;
            virial[2] = F.x*m_pos.z;
            virial[3] = F.y*m_pos.y;
            virial[4] = F.y*m_pos.z;
            virial[5] = F.z*m_pos.z;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("e_field");
            }
        #endif

    protected:
        Scalar3 m_pos;                //!< particle position
        BoxDim m_box;                 //!< box dimensions
        Scalar m_qi;                  //!< particle charge
        Scalar3 m_field;              //!< the field vector
   };


#endif // __EVALUATOR_EXTERNAL_LAMELLAR_H__
