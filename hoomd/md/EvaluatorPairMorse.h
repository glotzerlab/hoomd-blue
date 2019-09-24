// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_MORSE_H__
#define __PAIR_EVALUATOR_MORSE_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairMorse.h
    \brief Defines the pair evaluator class for Morse potential
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Morse pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ.

    <b>Morse specifics</b>

    EvaluatorPairMorse evaluates the function:
    \f[ V_{\mathrm{Morse}}(r) = D_0 \left[ \exp \left(-2\alpha \left(r - r_0\right) \right)
                                           -2\exp \left(-\alpha \left(r-r_0\right) \right)  \right] \f]

    Morse potential does not need diameter or charge. Three parameters are specified and stored in a Scalar4, for speed.
    \a \f[ D_0 \f] is placed in \a params.x, \a \f[ \alpha \f] is in \a params.y, and \f[ r_0 \f] is in \a params.z.
    \a param.w is always set to zero, and is ignored.

*/
class EvaluatorPairMorse
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar4 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairMorse(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), D0(_params.x), alpha(_params.y), r0(_params.z)
            {
            }

        //! Morse doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Morse doesn't use charge
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional diameter values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed
                  in PotentialPair.

            \return True if they are evaluated or false if they are not because we are beyond the cutoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq)
                {
                Scalar r = fast::sqrt(rsq);
                Scalar Exp_factor = fast::exp(-alpha*(r-r0));

                pair_eng = D0 * Exp_factor * (Exp_factor - Scalar(2.0));
                force_divr = Scalar(2.0) * D0 * alpha * Exp_factor * (Exp_factor - Scalar(1.0)) / r;

                if (energy_shift)
                    {
                    Scalar rcut = fast::sqrt(rcutsq);
                    Scalar Exp_factor_cut = fast::exp(-alpha*(rcut-r0));
                    pair_eng -= D0 * Exp_factor_cut * (Exp_factor_cut - Scalar(2.0));
                    }
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
            return std::string("morse");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar D0;      //!< Depth of the Morse potential at its minimum
        Scalar alpha;   //!< Controls width of the potential well
        Scalar r0;      //!< Offset, i.e., position of the potential minimum
    };


#endif // __PAIR_EVALUATOR_MORSE_H__
