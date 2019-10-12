// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jproc

#ifndef __PAIR_EVALUATOR_LJGAUSS_H__
#define __PAIR_EVALUATOR_LJGAUSS_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairLJGauss.h
    \brief Defines the pair evaluator class for Lennard Jones Gaussian potentials
    \details .....
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

    <b>LJ Gauss specifics</b>

    EvaluatorPairLJGauss evaluates the function:
    \f V_{\mathrm{gauss}}(r) = \frac{1}{r^{12}} - \frac{2}{r^{6}} - \epsilon e^{- \frac{\left(r - r_{0}\right)^{2}}{2 \sigma^{2}}} \f]
    This implementation contains a normalization term.

    The LJ Gaussian potential does not need diameter or charge. Three parameters are specified and stored in a Scalar3.
    \a epsilon is placed in \a params.x, \a sigma^2 is in \a params.y, and \a r_0 in \a params.z.

*/
class EvaluatorPairLJGauss
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar3 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairLJGauss(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), epsilon(_params.x), sigma2(_params.y), r0(_params.z)
            {
            }

        //! LJGauss doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! LJGauss doesn't use charge
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

            \return True if they are evaluated or false if they are not because we are beyond the cuttoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq)
                {
                const Scalar sqrt_2pi= Scalar(2.0) * M_SQRT2 / M_2_SQRTPI;
                const Scalar norm_const = Scalar(12) * pow(2.0,5.0/6.0) / 55;
                Scalar r = fast::sqrt(rsq);
                Scalar rdiff = r - r0;
                Scalar rdiff_sigma2 = rdiff / sigma2;
                Scalar exp_val = fast::exp(-Scalar(0.5) * rdiff_sigma2 * rdiff);
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r6inv = r2inv * r2inv * r2inv;

                force_divr = (r2inv * r6inv * Scalar(12.0) * (r6inv - Scalar(1.0))) - (exp_val * epsilon * rdiff_sigma2 / r);
                pair_eng = r6inv * (r6inv - Scalar(2.0)) - exp_val * epsilon;

                if (energy_shift)
                    {
                    Scalar rcut2inv = Scalar(1.0)/rcutsq;
                    Scalar rcut6inv = rcut2inv * rcut2inv * rcut2inv;
                    pair_eng -= rcut6inv * (rcut6inv - Scalar(2.0)) - (epsilon * fast::exp(-Scalar(1.0)/Scalar(2.0) * (rcutsq - r0) / sigma2));
                    }

                Scalar sigma = fast::sqrt(sigma2);
                Scalar norm_term=Scalar(1.0)/(norm_const+ sqrt_2pi * epsilon * sigma);
                force_divr *= norm_term;
                pair_eng *= norm_term;

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
            return std::string("lj_gauss");
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
        Scalar sigma2;   //!< sigma^2 parameter extracted from the params passed to the constructor
        Scalar r0;       //!< r0 prarameter extracted from the params passed to the constructor
    };


#endif // __PAIR_EVALUATOR_LJGAUSS_H__
