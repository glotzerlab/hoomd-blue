// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: zhoupj

#ifndef __PAIR_EVALUATOR_FOURIER_H__
#define __PAIR_EVALUATOR_FOURIER_H__

#ifndef NVCC
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairFourier.h
    \brief Defines the pair evaluator class for potential in form of Fourier series

    \details .....
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Fourier pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ.

    <b>Fourier specifics</b>

    EvaluatorPairFourier evaluates the function:
    \f[ V_{\mathrm{Fourier}}(r) = \frac{1}{r^{12}}
    + \frac{1}{r^2}\sum_{n=1}^4 [a_n cos(\frac{n \pi r}{r_{cut}})
    + b_n sin(\frac{n \pi r}{r_{cut}})] \f]

    where:
    \f[ a_1 = \sum_{n=2}^4 (-1)^n a_n cos(\frac{n \pi r}{r_{cut}}) \f]

    \f[ b_1 = \sum_{n=2}^4 n (-1)^n b_n cos(\frac{n \pi r}{r_{cut}}) \f]

    is calculated to enforce close to zero value at r_cut

    The Fourier potential does not need diameter or charge. two sets of parameters: a and b (both list of size 3) are specified and stored in a pair_fourier_params type.
    - \a a is placed in params.a,
    - \a b is placed in params.b.

*/
struct pair_fourier_params
{
  Scalar a[3];      //!< Fourier component coefficents
  Scalar b[3];      //!< Fourier component coefficents
};

class EvaluatorPairFourier
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef pair_fourier_params param_type; //first try a 4th order fourier expression of potential
        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */

        DEVICE EvaluatorPairFourier(Scalar _rsq, Scalar _rcutsq,
                                    const param_type& _params) : rsq(_rsq), rcutsq(_rcutsq), params(_params)
            {
            }

        //! Fourier doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Fourier doesn't use charge
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
        DEVICE bool evalForceAndEnergy(Scalar& force_divr,
                                       Scalar& pair_eng,
                                       bool energy_shift)
            {
            // compute the force divided by r in force_divr
            if (rsq < rcutsq)
                {
                Scalar half_period = fast::sqrt(rcutsq);
		Scalar period_scale = M_PI / half_period;
                Scalar r = fast::sqrt(rsq);
                Scalar x = r * period_scale;
                Scalar r1inv = Scalar(1)/r;
                Scalar r2inv = Scalar(1)/rsq;
                Scalar r3inv = r1inv * r2inv;
                Scalar r12inv = r3inv * r3inv * r3inv * r3inv;
                Scalar a1 = 0;
                Scalar b1 = 0;
                for (int i=2; i<5; i++)
                    {
                    a1 = a1 + fast::pow(Scalar(-1),Scalar(i)) * params.a[i-2];
                    b1 = b1 + i * fast::pow(Scalar(-1),Scalar(i)) * params.b[i-2];
                    }
                Scalar theta = x;
                Scalar s;
                Scalar c;
                fast::sincos(theta, s, c);
                Scalar fourier_part = a1 * c + b1 * s;
                force_divr = a1 * s - b1 * c;

                for (int i=2; i<5; i++)
                    {
                    theta = Scalar(i) * x;
                    fast::sincos(theta, s, c);
                    fourier_part += params.a[i-2] * c + params.b[i-2] * s;
                    force_divr += params.a[i-2] * Scalar(i) * s - params.b[i-2] * Scalar(i) * c;
                    }

                force_divr = r1inv * (r1inv * r12inv * Scalar(12)
                           + r2inv * period_scale * force_divr
                           + Scalar(2) * r3inv * fourier_part);
                pair_eng = r12inv + r2inv * fourier_part;

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
            return std::string("fourier");
            }

        std::string getShapeSpec() const
            {
            throw std::runtime_error("Shape definition not supported for this pair potential.");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        const pair_fourier_params& params;      //!< Fourier component coefficents
    };


#endif // __PAIR_EVALUATOR_FOURIER_H__
