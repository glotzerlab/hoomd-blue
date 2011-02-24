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

// $Id$
// $URL$
// Maintainer: grva

#ifndef __EvaluatorPairAniso__
#define __EvaluatorPairAniso__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"
#include "DirectionalEvaluatorPair.h"

/*! \file EvaluatorPairAniso.h
    \brief Defines a mostly abstract base class for anisotropic pair potentials
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

//! Class for evaluating anisotropic pair interations
/*! <b>General Overview</b>

    Provides a base class for detailed anisotropic pairwise interactions
*/
template <typename isoEval>
class EvaluatorPairAniso
    {
    public:
        //! Constructs the pair potential evaluator
        /*! \param _iEv a pair evaluator for the isotropic part of the interaction
            \param _dEv a DirectionalEvaluatorPair for the directional part
        */
        DEVICE EvaluatorPairAniso(isoEval _iEv, DirectionalEvaluatorPair _dEv)
            : iEv(_iEv), dEv(_dEv)
            {
            }
        
        //! uses diameter
        DEVICE static bool needsDiameter()
            {
            return (iEv.needsDiameter() || dEv.needsDiameter());
            }

        //! Accept the optional diameter values
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE void setDiameter(Scalar di, Scalar dj)
            {
            if (params.iEv.needsDiameter())
                iEv.setDiameter(di,dj);
            if (params.dEv.needsDiameter())
                dEv.setDiameter(di,dj);
            }


        //! whether pair potential requires charges
        //! This function is pure virtual
        DEVICE static bool needsCharge()
            {
            return (iEv.needsCharge() || dEv.needsCharge());
            }

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE void setCharge(Scalar qi, Scalar qj)
            {
            if (params.iEv.needsCharge())
                iEv.setCharge(di,dj);
            if (params.dEv.needsCharge())
                dEv.setCharge(di,dj);
            }
        
        //! Evaluate the force and energy
        //! This function is pure virtual
        /*! \param force Output parameter to write the computed force.
            \param pair_eng Output parameter to write the computed pair energy.
            \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the cutoff.
            \param torque_i The torque exterted on the i^th particle.
            \param torque_j The torque exterted on the j^th particle.
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
                  in PotentialPair.
            
            \return True if they are evaluated or false if they are not because we are beyond the cutoff.
        */
        DEVICE bool evalPair(Scalar3& force, Scalar& pair_eng, bool energy_shift,
			Scalar3& torque_i, Scalar3& torque_j, const Scalar3& dr)
            {
            Scalar force_divr;
            if ( iEv.evalForceAndEnergy(force_divr,pair_eng,energy_shift) )
                {
                // variables to use in the following;
                Scalar3 dforce;
                Scalar isoModulator;

                // evaluate the directional part of the pair interaction
                dEv.evalPair(dforce,isoModulator,torque_i,torque_j);

                // compute force
                force.x = isoModulator*force_divr*dr.x + pair_eng*dforce.x;
                force.y = isoModulator*force_divr*dr.y + pair_eng*dforce.y;
                force.z = isoModulator*force_divr*dr.z + pair_eng*dforce.z;

                // correct the torques with the isotropic prefactor
		// first the i^th ...
                torque_i.x *= pair_eng;
                torque_i.y *= pair_eng;
                torque_i.z *= pair_eng;

		// ... then the j^th
                torque_j.x *= pair_eng;
                torque_j.y *= pair_eng;
                torque_j.z *= pair_eng;

                // now correct the pair energy by the anisotropic modulator
                // N.B. order is important, can't do this before above steps
                pair_eng *= isoModulator;

                return true;
                }
            else
                {
                return false;
                }

            }
        
        #ifndef NVCC
        //! Get the name of the potential
        //! This function is pure virtual
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        virtual static std::string getName() = 0;
        #endif

    protected:
        isoEval iEv;     //!< An isotropic pair evaluator
        DirectionalEvaluatorPair dEv;     //!< A directional pair evaluator
    };


#endif // __EvaluatorPairAniso__

