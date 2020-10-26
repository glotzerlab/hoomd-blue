// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __PAIR_EVALUATOR_FORCE_SHIFTED_LJ_H__
#define __PAIR_EVALUATOR_FORCE_SHIFTED_LJ_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairForceShiftedLJ.h
    \brief Defines the pair evaluator class for LJ potentials
    \details As the prototypical example of a MD pair potential, this also serves as the primary documentation and
    base reference for the implementation of pair evaluators.
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the force shifted LJ pair potential
/*! This evaluator is a variant of the Lennard-Jones pair potential, for which the force goes smoothly
    to zero at \f$ r = r_{\mathrm{cut}} \f$.

    EvaluatorPairForceShiftedLJ evaluates the function:
    \f[ V(r) = V_{LJ}(r) + \Delta V(r) \f], where
    \f[ V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
                                            \alpha \left( \frac{\sigma}{r} \right)^{6} \right] \f]
    is the standard Lennard-Jones pair potential and
    \f[ \Delta V(r) = -(r - r_{\mathrm{cut}}) \frac{\partial V_{\mathrm{LJ}}}{\partial r}(r_{\mathrm{cut}}) \f].
    The constant value \f$ -\frac{1}{r} \frac{\partial V}{\partial r}(r_{\mathrm{cut}}) \f$ is
        subtracted from \a force_divr .

    The two parameters \a lj1 and \a lj2 are the same as for the (non-force-shifted) LJ potential.

*/
class EvaluatorPairForceShiftedLJ
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential and the force go to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairForceShiftedLJ(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), lj1(_params.x), lj2(_params.y)
            {
            }

        //! LJ doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJ doesn't use charge
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
            if (rsq < rcutsq && lj1 != 0)
                {
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;
                force_divr= r2inv * r6inv * (Scalar(12.0)*lj1*r6inv - Scalar(6.0)*lj2);

                pair_eng = r6inv * (lj1*r6inv - lj2);

                Scalar rcut2inv = Scalar(1.0)/rcutsq;
                Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;

                if (energy_shift)
                    pair_eng -= rcut6inv * (lj1*rcut6inv - lj2);

                // shift force and add linear term to potential
                Scalar rcut_r_inv = fast::rsqrt(rsq*rcutsq);
                Scalar force_rcut_at_rcut = rcut6inv * (Scalar(12.0)*lj1*rcut6inv - Scalar(6.0)*lj2);
                force_divr -= rcut_r_inv * force_rcut_at_rcut;
                pair_eng += (rsq*rcut_r_inv-Scalar(1.0))*force_rcut_at_rcut;

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
            return std::string("force_shift_lj");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar lj1;     //!< lj1 parameter extracted from the params passed to the constructor
        Scalar lj2;     //!< lj2 parameter extracted from the params passed to the constructor
    };


#endif // __PAIR_EVALUATOR_FORCE_SHIFTED_LJ_H__
