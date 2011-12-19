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

#ifndef __BOND_EVALUATOR_HARMONIC_DPD_H__
#define __BOND_EVALUATOR_HARMONIC_DPD_H__

#ifndef NVCC
#include <string>
#endif

#include <hoomd/HOOMDMath.h>

/*! \file EvaluatorBondHarmonicDPD.h
    \brief Defines the bond evaluator class for composite harmonic bond
           + DPD pair potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
//! DEVICE is __host__ __device__ when included in nvcc and blank when included into the host compiler
#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif

// call different optimized sqrt functions on the host / device
//! RSQRT is rsqrtf when included in nvcc and 1.0 / sqrt(x) when included into the host compiler
#ifdef NVCC
#define RSQRT(x) rsqrtf( (x) )
#else
#define RSQRT(x) Scalar(1.0) / sqrt( (x) )
#endif

//! Class for evaluating the harmonic bond potential
class EvaluatorBondHarmonicDPD
    {
    public:
        //! Define the parameter type used by this bond potential evaluator
        typedef Scalar4 param_type;

        //! Constructs the bond potential evaluator
        /*! \param _rsq Squared distance beteen the particles
            \param _params Per bond type parameters of this potential
        */
        DEVICE EvaluatorBondHarmonicDPD(Scalar _rsq, const param_type& _params)
            : rsq(_rsq),K(_params.x), r_0(_params.y), rcut(_params.z),
              a(_params.w)
            {
            }

        //! Harmonic doesn't use diameter
        DEVICE static bool needsDiameter() { return false; }

        //! Accept the optional diameter values
        /*! \param da Diameter of particle a
            \param db Diameter of particle b
        */
        DEVICE void setDiameter(Scalar da, Scalar db) { }

        //! Harmonic doesn't use charge
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
            // evaluate harmonic bond potential
            Scalar r = sqrt(rsq);
            force_divr = K * (r_0 / r - Scalar(1.0));
            if (!isfinite(force_divr)) return false;
            bond_eng = Scalar(0.5) * K * (r_0 - r) * (r_0 - r);

            if (rsq < rcut*rcut)
                {
                // evaluate DPD pair potential
                Scalar rinv = RSQRT(rsq);
                Scalar r = Scalar(1.0) / rinv;
                Scalar rcutinv = Scalar(1.0)/ rcut;

                force_divr += a*(rinv - rcutinv);
                bond_eng += a * (rcut - r) - Scalar(1.0/2.0) * a * rcutinv * (rcut*rcut - rsq);
                }

            return true;
            }

        #ifndef NVCC
        //! Get the name of this potential
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName()
            {
            return std::string("harmonic+dpd");
            }
        #endif

    protected:
        Scalar rsq;        //!< Stored rsq from the constructor
        Scalar K;          //!< K parameter
        Scalar r_0;        //!< r_0 parameter
        Scalar rcut;       //!< Cutoff for pair potential
        Scalar a;           //!< a parameter for pair potential
    };


#endif // __BOND_EVALUATOR_HARMONIC_DPD_H__
