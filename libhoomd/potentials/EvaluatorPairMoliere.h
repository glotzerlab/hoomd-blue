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

// $Id: EvaluatorPairLJ.h 2862 2010-03-12 19:16:16Z joaander $
// $URL: http://codeblue.umich.edu/hoomd-blue/svn/trunk/libhoomd/potentials/EvaluatorPairLJ.h $
// Maintainer: joaander

#ifndef __PAIR_EVALUATOR_MOLIERE__
#define __PAIR_EVALUATOR_MOLIERE__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

/*! \file EvaluatorPairMoliere.h
    \brief Defines the pair evaluator class for Moliere potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Moliere pair potential.
/*! EvaluatorPairMoliere evaluates the function
    \f[ V_{\mathrm{Moliere}}(r) = \frac{Z_i Z_j e^2}{4 \pi \varepsilon_0 r_{ij}} \left[ 0.35 \exp \left(-0.3 \frac{r_{ij}}{a_F} \right) +
                                    0.55 \exp \left( -1.2 \frac{r_{ij}}{a_F} \right) +
                                    0.10 \exp \left( -6.0 \frac{r_{ij}}{a_F} \right) \right] \f]

    where
    \f[ a_F = \frac{0.8853 a_0}{ \left( \sqrt{Z_i} + \sqrt{Z_j} \right)^{2/3}} \f]

    and \a a_0 is the Bohr radius and \a Z_x denotes the atomic number of species \a x.

*/

class EvaluatorPairMoliere
{
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance between the particles.
            \param _rcutsq Squared distance at which the potential goes to zero.
            \param _params Per type-pair parameters of this potential
        */
        DEVICE EvaluatorPairMoliere(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), Zsq(_params.x), aF(_params.y)
            {
            }

        //! Moliere potential does not use particle diameters.
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! Moliere potential does not use particles charges
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
                Scalar exp1 = Scalar(0.35) * fast::exp( Scalar(-0.3) / aF / rinv );
                Scalar exp2 = Scalar(0.55) * fast::exp( Scalar(-1.2) / aF / rinv );
                Scalar exp3 = Scalar(0.1) * fast::exp( Scalar(-6.0) / aF / rinv );

                // evaluate the force
                force_divr = rinv * ( exp1 + exp2 + exp3 );
                force_divr += Scalar(1.0) / aF * ( Scalar(0.3) * exp1 + Scalar(1.2) * exp2 + Scalar(6.0) * exp3 );
                force_divr *= Zsq * r2inv;

                // evaluate the pair energy
                pair_eng = Zsq * rinv * ( exp1 + exp2 + exp3 );
                if (energy_shift)
                {
                    Scalar rcutinv = fast::rsqrt(rcutsq);

                    Scalar expcut1 = Scalar(0.35) * fast::exp( Scalar(-0.3) / aF / rcutinv );
                    Scalar expcut2 = Scalar(0.55) * fast::exp( Scalar(-1.2) / aF / rcutinv );
                    Scalar expcut3 = Scalar(0.1) * fast::exp( Scalar(-6.0) / aF / rcutinv);

                    pair_eng -= Zsq * rcutinv * ( expcut1 + expcut2 + expcut3 );
                }

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
            return std::string("moliere");
        }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar Zsq;     //!< Zsq parameter extracted from the params passed to the constructor
        Scalar aF;      //!< aF parameter extracted from the params passed to the constructor
};

#endif // __PAIR_EVALUATOR_MOLIERE__
