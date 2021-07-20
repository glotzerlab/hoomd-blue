// Copyright (c) 2009-2019 The Regents of the Fudan University
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


//Maintainer Xiaotian Li

#ifndef __EVALUATOR_EXTERNAL_GRAVITATIONAL_FIELD_H__
#define __EVALUATOR_EXTERNAL_GRAVITATIONAL_FIELD_H__

#ifndef NVCC
#include <string>
#endif

#include <math.h>
#include "hoomd/HOOMDMath.h"
#include "hoomd/BoxDim.h"

/*! \file EvaluatorExternalGravitationalField.h
    \brief Defines the external potential evaluator to induce a periodic ordered phase
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating an gravitational field
/*! <b>General Overview</b>
    The external potential \f$V(\vec{r}) \f$ is implemented using the following formula:

    \f[
    V(\vec{r}) = - m_i \vec{g} \cdot \vec{r}
    \f]

    where \f$E\f$ is the strength of the gravitational field and \f$m_i\f$ is the mass of particle i.
*/
class EvaluatorExternalGravitationalField
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
        DEVICE EvaluatorExternalGravitationalField(Scalar3 X, const BoxDim& box, const param_type& params, const field_type& field)
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
        DEVICE static bool needsMass() { return true; }
        //! Accept the optional diameter value
        /*! \param mi Mass of particle i
        */
        DEVICE void setMass(Scalar qi) { m_mi = mi; }

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
            F = m_mi * m_field;
            energy = -m_mi * dot(m_field,m_pos);

            virial[0] = F.x*m_pos.x;//Fx * rx
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
            return std::string("g_field");
            }
        #endif

    protected:
        Scalar3 m_pos;                //!< particle position
        BoxDim m_box;                 //!< box dimensions
        Scalar m_mi;                  //!< particle mass
        Scalar3 m_field;              //!< the field vector
   };


#endif // __EVALUATOR_EXTERNAL_GRAVITATIONAL_FIELD_H__
