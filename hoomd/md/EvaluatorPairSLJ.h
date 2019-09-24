// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_SLJ_H__
#define __PAIR_EVALUATOR_SLJ_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairSLJ.h
    \brief Defines the pair evaluator class for shifted Lennard-Jones potentials
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

    <b>SLJ specifics</b>

    EvaluatorPairSLJ evaluates the function:
    \f{eqnarray*}
    V_{\mathrm{SLJ}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r - \Delta} \right)^{12} -
                           \left( \frac{\sigma}{r - \Delta} \right)^{6} \right] & r < (r_{\mathrm{cut}} + \Delta) \\
                         = & 0 & r \ge (r_{\mathrm{cut}} + \Delta) \\
    \f}
    where \f$ \Delta = (d_i + d_j)/2 - 1 \f$ and \f$ d_i \f$ is the diameter of particle \f$ i \f$.

    The SLJ potential does not need charge, but does need diameter. Two parameters are specified and stored in a
    Scalar2. \a lj1 is placed in \a params.x and \a lj2 is in \a params.y.

    These are related to the standard lj parameters sigma and epsilon by:
    - \a lj1 = 4.0 * epsilon * pow(sigma,12.0)
    - \a lj2 = 4.0 * epsilon * pow(sigma,6.0);

    Due to the way that SLJ modifies the cutoff condition, it will not function properly with the xplor shifting mode.
*/
class EvaluatorPairSLJ
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles
            \param _rcutsq Squared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairSLJ(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), lj1(_params.x), lj2(_params.y)
            {
            }

        //! SLJ uses diameter
        DEVICE static bool needsDiameter() { return true; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj)
            {
            delta = (di + dj) / Scalar(2.0) - Scalar(1.0);
            }

        //! SLJ doesn't use charge
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
            // precompute some quantities
            Scalar rinv = fast::rsqrt(rsq);
            Scalar r = Scalar(1.0) / rinv;
            Scalar rcutinv = fast::rsqrt(rcutsq);
            Scalar rcut = Scalar(1.0) / rcutinv;

            // compute the force divided by r in force_divr
            if (r < (rcut + delta) && lj1 != 0)
                {
                Scalar rmd = r - delta;
                Scalar rmdinv = Scalar(1.0) / rmd;
                Scalar rmd2inv = rmdinv * rmdinv;
                Scalar rmd6inv = rmd2inv * rmd2inv * rmd2inv;
                force_divr= rinv * rmdinv * rmd6inv * (Scalar(12.0)*lj1*rmd6inv - Scalar(6.0)*lj2);

                pair_eng = rmd6inv * (lj1*rmd6inv - lj2);

                if (energy_shift)
                    {
                    Scalar rcut2inv = rcutinv * rcutinv;
                    Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                    pair_eng -= rcut6inv * (lj1*rcut6inv - lj2);
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
            return std::string("slj");
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
        Scalar delta;   //!< Delta parameter extracted from the call to setDiameter
    };

#endif // __PAIR_EVALUATOR_SLJ_H__
