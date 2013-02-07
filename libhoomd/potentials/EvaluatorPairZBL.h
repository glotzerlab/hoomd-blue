/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef __PAIR_EVALUATOR_ZBL__
#define __PAIR_EVALUATOR_ZBL__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

/*! \file EvaluatorPairZBL.h
    \brief Defines the pair evaluator class for ZBL potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// call different optimized exp functions on the host / device
// EXP is expf when included in nvcc and exp when included in the host compiler
#if defined NVCC && defined SINGLE_PRECISION
#define EXP expf
#else
#define EXP exp
#endif

// call different optimized sqrt functions on the host / device
// SQRT is sqrtf when included in nvcc and sqrt when included in the host compiler
#if defined NVCC && defined SINGLE_PRECISION
#define SQRT sqrtf
#else
#define SQRT sqrt
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
                Scalar rinv = RSQRT(rsq);

                // precalculate the exponential terms
                Scalar exp1 = Scalar(0.1818) * EXP( Scalar(-3.2) / aF / rinv );
                Scalar exp2 = Scalar(0.5099) * EXP( Scalar(-0.9423) / aF / rinv );
                Scalar exp3 = Scalar(0.2802) * EXP( Scalar(-0.4029) / aF / rinv );
                Scalar exp4 = Scalar(0.02817) * EXP( Scalar(-0.2016) / aF / rinv );

                // evaluate the force
                force_divr = rinv * ( exp1 + exp2 + exp3 + exp3 );
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
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar Zsq;     //!< Zsq parameter extracted from the params passed to the constructor
        Scalar aF;      //!< aF parameter extracted from the params passed to the constructor
};

#endif // __PAIR_EVALUATOR_ZBL__
