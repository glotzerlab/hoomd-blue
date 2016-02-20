/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

#ifndef __PAIR_EVALUATOR_REACTION_FIELD_H__
#define __PAIR_EVALUATOR_REACTION_FIELD_H__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

/*! \file EvaluatorPairReactionField.h
    \brief Defines the pair evaluator class for ReactionField potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the Onsager reaction field pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ

    <b>ReactionField specifics</b>

    EvaluatorPairReactionField evaluates the function:
    \f[ V_{\mathrm{RF}}(r) = \varepsilon \left[ \frac{1}{r} +
        \frac{(\epsilon_{RF}-1) r^2}{(2 \epsilon_{RF} + 1) r_c^3} \right]\f]

    The reaction field potential does not require charge or diameter. Two parameters,
    \f$ \varepsilon \f$ and \f$ \epsilon_{RF} \f$ are needed.

    \a \varepsilon is placed in \a params.x and \a \epsilon_{RF} is in \a params.y.

    If \epsilon_{RF} is zero, it will be treated as infinity.
*/
class EvaluatorPairReactionField
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar2 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _rcutsq Sqauared distance at which the potential goes to 0
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairReactionField(Scalar _rsq, Scalar _rcutsq, const param_type& _params)
            : rsq(_rsq), rcutsq(_rcutsq), epsilon(_params.x), epsrf(_params.y)
            {
            }

        //! ReactionField doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }
        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj) { }

        //! ReactionField doesn't use charge
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
            if (rsq < rcutsq && epsilon != 0)
                {
                Scalar rcut3inv = fast::rsqrt(rcutsq)/rcutsq;
                Scalar rinv = fast::rsqrt(rsq);
                Scalar r = Scalar(1.0) / rinv;
                Scalar r2inv = Scalar(1.0) / rsq;

                Scalar eps_fac = (epsrf - Scalar(1.0))/(Scalar(2.0)*epsrf+Scalar(1.0))*rcut3inv;
                if (epsrf == Scalar(0.0))
                    {
                    eps_fac = Scalar(1.0/2.0)*rcut3inv;
                    }

                force_divr = epsilon * (r2inv * rinv - Scalar(2.0)*eps_fac);
                pair_eng = epsilon * (rinv + eps_fac*r*r);

                if (energy_shift)
                    {
                    Scalar rcutinv = fast::rsqrt(rcutsq);
                    Scalar rcut = Scalar(1.0) / rcutinv;
                    pair_eng -= epsilon * (rcutinv + eps_fac*rcut*rcut);
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
            return std::string("reaction_field");
            }
        #endif

    protected:
        Scalar rsq;     //!< Stored rsq from the constructor
        Scalar rcutsq;  //!< Stored rcutsq from the constructor
        Scalar epsilon; //!< epsilon parameter extracted from the params passed to the constructor
        Scalar epsrf;   //!< epsilon_rf parameter extracted from the params passed to the constructor
    };


#endif // __PAIR_EVALUATOR_REACTION_FIELD_H__
