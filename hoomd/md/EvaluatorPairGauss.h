// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_GAUSS_H__
#define __PAIR_EVALUATOR_GAUSS_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairGauss.h
    \brief Defines the pair evaluator class for Gaussian potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Gaussian pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ

    <b>Gauss specifics</b>

    EvaluatorPairGauss evaluates the function:
    \f[ V_{\mathrm{gauss}}(r) = \varepsilon \exp \left[ -\frac{1}{2}\left( \frac{r}{\sigma} \right)^2 \right] \f]

    The Gaussian potential does not need diameter or charge. Two parameters are specified and stored in a Scalar2.
    \a epsilon is placed in \a params.x and \a sigma is in \a params.y.

    These are related to the standard lj parameters sigma and epsilon by:
    - \a epsilon = \f$ \varepsilon \f$
    - \a sigma = \f$ \sigma \f$

*/
class EvaluatorPairGauss
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairGauss(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), epsilon(_params.x), sigma(_params.y)
            {
            }

        //! Gauss doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Gauss doesn't use charge
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
                Scalar sigma_sq = sigma*sigma;
                Scalar r_over_sigma_sq = rsq / sigma_sq;
                Scalar exp_val = fast::exp(-Scalar(1.0)/Scalar(2.0) * r_over_sigma_sq);

                force_divr = epsilon / sigma_sq * exp_val;
                pair_eng = epsilon * exp_val;

                if (energy_shift)
                    {
                    pair_eng -= epsilon * fast::exp(-Scalar(1.0)/Scalar(2.0) * rcutsq / sigma_sq);
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
            return std::string("gauss");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar epsilon; //!< epsilon parameter extracted from the params passed to the constructor
        Scalar sigma;   //!< sigma parameter extracted from the params passed to the constructor
    };


#endif // __PAIR_EVALUATOR_GAUSS_H__
