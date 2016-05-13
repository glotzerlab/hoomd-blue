// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: sbarr

#ifndef __PAIR_EVALUATOR_EWALD_H__
#define __PAIR_EVALUATOR_EWALD_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairEwald.h
    \brief Defines the pair evaluator class for Ewald potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#if defined NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Ewald pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ

    <b>Ewald specifics</b>

    EvaluatorPairEwald evaluates the function:

    \f[ V_{\mathrm{Ewald}}(r) = q_i q_j erfc(\kappa r)/r \f]

    The Ewald potential does not need diameter. One parameters is specified and stored in a Scalar.
    \a kappa is placed in \a params
*/
class EvaluatorPairEwald
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairEwald(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
          : rsq(_rsq), rcutsq(_rcutsq), kappa(_params)
            {
            }

        //! Ewald doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Ewald uses charge !!!
        DEVICE static bool needsCharge() { return true; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj)
            {
            qiqj = qi * qj;
            }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
                  in PotentialPair.

            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            if (rsq < rcutsq && qiqj != 0)
                {
                Scalar rinv = fast::rsqrt(rsq);
                Scalar r = Scalar(1.0) / rinv;
                Scalar r2inv = Scalar(1.0) / rsq;

                Scalar erfc_by_r_val = fast::erfc(kappa * r) * rinv;

                force_divr = qiqj * r2inv * (erfc_by_r_val + Scalar(2.0)*kappa*fast::rsqrt(Scalar(M_PI)) * fast::exp(-kappa*kappa* rsq));
                pair_eng = qiqj * erfc_by_r_val ;

                return true;
                }
            else
                return false;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("ewald");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar kappa;   //!< kappa parameter extracted from the params passed to the constructor
        Scalar qiqj;    //!< product of qi and qj
    };


#endif // __PAIR_EVALUATOR_EWALD_H__
