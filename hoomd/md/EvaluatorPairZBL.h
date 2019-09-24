// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


#ifndef __PAIR_EVALUATOR_ZBL__
#define __PAIR_EVALUATOR_ZBL__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairZBL.h
    \brief Defines the pair evaluator class for ZBL potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the ZBL pair potential.
/*! EvaluatorPairZBL evaluates the function
    \f{eqnarray*}
        V_{\mathrm{ZBL}}(r) = & \frac{Z_i Z_j e^2}{4 \pi \epsilon_0 r_{ij}} \left[ 0.1818 \exp \left( -3.2 \frac{r_{ij}}{a_F} \right) + 0.5099 \exp \left( -0.9423 \frac{r_{ij}}{a_F} \right) + 0.2802 \exp \left( -0.4029 \frac{r_{ij}}{a_F} \right) + 0.02817 \exp \left( -0.2016 \frac{r_{ij}}{a_F} \right) \right], & r < r_{\mathrm{cut}} \\
                            = & 0, & r > r_{\mathrm{cut}} \\
    \f}

    where
    \f[ a_F = \frac{0.8853 a_0}{ \left( Z_i^{0.23} + Z_j^{0.23} \right) } \f]

    and \a a_0 is the Bohr radius and \a Z_x denotes the atomic number of species \a x.

*/

class EvaluatorPairZBL
{
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles.
            \param _rcutsq Squared distance at which the potential goes to zero.
            \param _params Per type-pair parameters of this potential
        */
        DEVICE EvaluatorPairZBL(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), Zsq(_params.x), aF(_params.y)
            {
            }

        //! ZBL potential does not use particle diameters.
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! ZBL potential does not use particle charges
        DEVICE static bool needsCharge() { return false; }
        //! Accept the optional charge values
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj) { }

        //! Evaluate the force and energy.
        /*! \param force_divr Output parameter to write the computed force divided by r
            \param pair_eng Output parameter to write the computed pair energy
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff

            \return True if they are evaluated or false if they are not because we are beyond the cutoff
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& pair_eng, bool energy_shift)
        {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq && Zsq != 0 && aF != 0)
            {
                Scalar r2inv = Scalar(1.0) / rsq;
                Scalar rinv = fast::rsqrt(rsq);

                // precalculate the exponential terms
                Scalar exp1 = Scalar(0.1818) * fast::exp( Scalar(-3.2) / aF / rinv );
                Scalar exp2 = Scalar(0.5099) * fast::exp( Scalar(-0.9423) / aF / rinv );
                Scalar exp3 = Scalar(0.2802) * fast::exp( Scalar(-0.4029) / aF / rinv );
                Scalar exp4 = Scalar(0.02817) * fast::exp( Scalar(-0.2016) / aF / rinv );

                // evaluate the force
                force_divr = rinv * ( exp1 + exp2 + exp3 + exp4 );
                force_divr += Scalar(1.0) / aF * ( Scalar(3.2) * exp1 \
                            + Scalar(0.9423) * exp2 + Scalar(0.4029) * exp3 \
                            + Scalar(0.2016) * exp4 );
                force_divr *= Zsq * r2inv;

                // evaluate the pair energy
                pair_eng = Zsq * rinv * ( exp1 + exp2 + exp3 + exp4 );

                return true;
            }
            else
                return false;
        }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name.  Must be short and all lowercase, as this is the name
            energies will be logged as via analyze.log.
        */
        static std::string getName()
            {
            return std::string("zbl");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar Zsq;     //!< Zsq parameter extracted from the params passed to the constructor
        Scalar aF;      //!< aF parameter extracted from the params passed to the constructor
};

#endif // __PAIR_EVALUATOR_ZBL__
