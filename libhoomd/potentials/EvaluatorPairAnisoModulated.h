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

#ifndef __EvaluatorPairAnisoModulated__
#define __EvaluatorPairAnisoModulated__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

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

template<typename isoEval, typename dirEval>
struct EvaluatorPairAnisoModulatedParamStruct
    {
    typedef typename isoEval::params iParam;
    typedef typename dirEval::params dParam;

    iParam iP;
    dParam dP;
    }

//! Class for evaluating anisotropic pair interations
/*! <b>General Overview</b>

    Provides a base class for detailed anisotropic pairwise interactions
*/
template <typename isoEval, typename dirEval>
class EvaluatorPairAnisoModulated
    {
    public:
	typedef EvaluatorPairAnisoModulatedParamStruct<isoEval,dirEval> param_type;

        //! Constructs the pair potential evaluator
        /*! \param _dr Displacement vector between particle centres of mass
            \param _rcutsq Squared distance at which the potential goes to 0
	    \param _q_i Quaterion of i^th particle
	    \param _q_j Quaterion of j^th particle
            \param _params Per type pair parameters of this potential
        */
        DEVICE EvaluatorPairAnisoModulated(Scalar3 _dr, Scalar4 _quat_i, Scalar4 _quat_j, Scalar _rcutsq,
			param_type _params)
	    : dr(_dr)
            {
	    Scalar rsq = dr.x*dr.x+dr.y*dr.y+dr.z*dr.z;
	    isoEval(rsq,_rcutsq,_params.iP);
	    dirEval(_dr,_quat_i,_quat_j,_params.dP);
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
        DEVICE bool
		evalPair(Scalar3& force, Scalar& pair_eng, bool energy_shift, Scalar3& torque_i, Scalar3& torque_j)
            {
	    // used in computation
	    Scalar isoModulator,
	    Scalar force_divr;
	    
	    // do individual computes
            dEv.evalPair(force,isoModulator,torque_i,torque_j);
	    iEv.evalForceAndEnergy(force_divr,pair_eng,energy_shift);

	    // correct forces
	    force.x = pair_eng*force.x+dr.x*force_divr;
	    force.y = pair_eng*force.y+dr.y*force_divr;
	    force.z = pair_eng*force.z+dr.z*force_divr;

	    // correct torques
	    torque_i.x *= pair_eng;
	    torque_i.y *= pair_eng;
	    torque_i.z *= pair_eng;

	    torque_j.x *= pair_eng;
	    torque_j.y *= pair_eng;
	    torque_j.z *= pair_eng;

	    // correct pair energy
	    pair_eng *= isoModulator;
            }

        #ifndef NVCC
        //! Get the name of the potential
	//! This function is pure virtual
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        static std::string getName() { }
        #endif

    protected:
	Scalar3 dr;
        isoEval iEv;     //!< An isotropic pair evaluator
        dirEval dEv;     //!< A directional pair evaluator
    };


#endif // __EvaluatorPairAnisoModulated__

