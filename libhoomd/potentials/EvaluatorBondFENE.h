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

#ifndef __BOND_EVALUATOR_FENE_H__
#define __BOND_EVALUATOR_FENE_H__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

/*! \file EvaluatorBondFENE.h
    \brief Defines the bond evaluator class for FENE potentials

    The parameters are:
    - \a K (params.x) Stiffness parameter for the force computation
    - \a r_0 (params.y) maximum bond length for the force computation
    - \a lj1 (params.z) Value of lj1 = 4.0*epsilon*pow(sigma,12.0)
       of the WCA potential in the force calculation
    - \a lj2 (params.w) Value of lj2 = 4.0*epsilon*pow(sigma,6.0)
       of the WCA potential in the force calculation
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

//! Class for evaluating the FENE bond potential
class EvaluatorBondFENE
    {
    public:
        //! Define the parameter type used by this pair potential evaluator
        typedef Scalar4 param_type;

        //! Constructs the pair potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorBondFENE(Scalar _rsq, const param_type& _params)
            : rsq(_rsq), K(_params.x), r_0(_params.y), lj1(_params.z), lj2(_params.w)
            {
            }

        //! This evaluator uses diameter information
        DEVICE static bool needsDiameter() { return true; }

        //! Accept the optional diameter values
        /*! \param da Diameter of particle a
            \param db Diameter of particle b
        */
        DEVICE void setDiameter(Scalar da, Scalar db)
            {
            diameter_a = da;
            diameter_b = db;
            }

        //! FENE  doesn't use charge
        DEVICE static bool needsCharge() { return false; }

        //! Accept the optional charge values
        /*! \param qa Charge of particle a
            \param qb Charge of particle b
        */
        DEVICE void setCharge(Scalar qa, Scalar qb) { }

        //! Evaluate the force and energy
        /*! \param force_divr Output parameter to write the computed force divided by r.
            \param bond_eng Output parameter to write the computed bond energy

            \return True if they are evaluated or false if the bond
                    energy is divergent
        */
        DEVICE bool evalForceAndEnergy(Scalar& force_divr, Scalar& bond_eng)
            {
            Scalar rmdoverr = Scalar(1.0);

            // Correct the rsq for particles that are not unit in size.
            Scalar rtemp = sqrt(rsq) - diameter_a/2 - diameter_b/2 + Scalar(1.0);
            rmdoverr = rtemp/sqrt(rsq);
            rsq = rtemp*rtemp;

            // compute the force magnitude/r in forcemag_divr (FLOPS: 9)
            Scalar r2inv = Scalar(1.0)/rsq;
            Scalar r6inv = r2inv * r2inv * r2inv;

            Scalar WCAforcemag_divr = Scalar(0.0);
            Scalar pair_eng = Scalar(0.0);

            Scalar sigma6inv = lj2/lj1;
            Scalar epsilon = lj2*lj2/Scalar(4.0)/lj1;

            // add != 0.0f check to allow epsilon=0 FENE bonds to go to r=0
            if ((r6inv > sigma6inv/Scalar(2.0)) && (epsilon != Scalar(0.0)))     //wcalimit 2^(1/6))^6 sigma^6
            {
                WCAforcemag_divr = r2inv * r6inv * (Scalar(12.0)*lj1*r6inv - Scalar(6.0)*lj2);
                pair_eng = (r6inv * (lj1*r6inv - lj2) + epsilon);

                if (!isfinite(pair_eng)) return false;
            }

            // Check if bond length restrictino is violated
            if (rsq >= r_0*r_0) return false;

            force_divr = -K / (Scalar(1.0) - rsq /
                         (r_0*r_0))*rmdoverr + WCAforcemag_divr*rmdoverr;
            bond_eng = -Scalar(0.5) * K * (r_0 * r_0) *
                           log(Scalar(1.0) - rsq/(r_0 * r_0));

            // non-finite results will be ignored in the force caclulation. This will result in the correct 0 force for r ~= 0. The energy
            // will be incorrect for r > r_0, however. Assuming that r > r_0 because K == 0, this is fine.
            if (!isfinite(force_divr)) return false;
            if (!isfinite(bond_eng)) return false;

            // add WCA pair energy
            bond_eng += pair_eng;

            return true;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("fene");
            }
        #endif

    protected:
        Scalar rsq;        //!< Stored rsq from the constructor
        Scalar K;          //!< K parameter
        Scalar r_0;        //!< r_0 parameter
        Scalar lj1;        //!< lj1 parameter
        Scalar lj2;        //!< lj2 parameter
        Scalar diameter_a; //!< diameter of particle A
        Scalar diameter_b; //!< diameter of particle B
    };


#endif // __BOND_EVALUATOR_FENE_H__
