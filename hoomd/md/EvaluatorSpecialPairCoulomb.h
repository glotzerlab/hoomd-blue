// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: unassigned

#ifndef __BOND_EVALUATOR_COULOMB_H__
#define __BOND_EVALUATOR_COULOMB_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorSpecialPairCoulomb.h
    \brief Defines the bond evaluator class for Coulomb interactions

    This is designed to be used e.g. for the scaled 1-4 interaction in all-atom force fields such as GROMOS or OPLS.
    It implements Coulombs law, as both LJ and Coulomb interactions are scaled by the same amount in the OPLS force
    field for 1-4 interactions. Thus it is intended to be used along with special_pair.LJ.
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Coulomb bond potential
/*! See the EvaluatorPairLJ class for the meaning of the parameters
 */
class EvaluatorSpecialPairCoulomb
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorSpecialPairCoulomb(Scalar _rsq, const param_type& _params)
            : rsq(_rsq), scale(_params.x), rcutsq(_params.y)
            {
            }

        //! Coulomb doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Coulomb use charge
        DEVICE static bool needsCharge() { return true; }

        //! Accept the optional charge values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj)
            {
            qiqj = qi * qj;
            }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param bond_eng Output parameter to write the computed bond energy

            \return True if they are evaluated or false if the bond
                    energy is not defined. Based on EvaluatorSpecialPairLJ which
                    returns true regardless whether it is evaluated or not.
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq && qiqj != 0)
                {
                Scalar r1inv = Scalar(1.0) / fast::sqrt(rsq);
                Scalar r2inv = Scalar(1.0) / rsq;
                Scalar r3inv = r2inv * r1inv;

                Scalar scaledQ = qiqj * scale;

                force_divr = scaledQ * r3inv;
                bond_eng = scaledQ * r1inv;
               }
            return true;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("coul");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar qiqj;    //!< product of charges from setCharge(qa, qb)
        Scalar scale;   //!< scaling factor to apply to Coulomb interaction
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
    };


#endif // __BOND_EVALUATOR_COULOMB_H__
