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

#ifndef __DirectionalEvaluatorPair__
#define __DirectionalEvaluatorPair__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

/*! \file DirectionalEvaluatorPair.h
    \brief Defines the pair evaluator class for Janus spheres
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

//! Class for evaluating the directional part of anisotropic pair interations
/*! <b>General Overview</b>

    Provides class for generalized diblock Janus spheres. It uses preexisting isotropic pair evaluators to do much of
    the work.
*/
template <typename param_type>
class DirectionalEvaluatorPair
    {
    public:
	//! Constructor, the only part of this class that isn't virtual
        DirectionalEvaluatorPair(param_type& _params)
        	: params(_params)
            {
            }
        
        //! uses diameter
        //! 
        DEVICE virtual static bool needsDiameter() = 0;

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param di Diameter of particle i
            \param dj Diameter of particle j
        */
        DEVICE virtual void setDiameter(Scalar di, Scalar dj) = 0;

        //! whether pair potential requires charges
        //! This function is pure virtual
        DEVICE virtual static bool needsCharge() = 0;

        //! Accept the optional diameter values
        //! This function is pure virtual
        /*! \param qi Charge of particle i
            \param qj Charge of particle j
        */
        DEVICE virtual void setCharge(Scalar qi, Scalar qj) = 0;
        
        //! Evaluate the force and energy
        //! This function is pure virtual
        /*! \param force Output parameter to write the computed force.
            \param isoModulator Output parameter to write the amount of modulation of the isotropic part
            \param torque_i The torque exterted on the i^th particle.
            \param torque_j The torque exterted on the j^th particle.
            \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are performed 
                  in PotentialPair.
            
            \return Always true
        */
        DEVICE virtual bool
        	evalPair(Scalar3& force, Scalar& isoModulator, Scalar3& torque_i, Scalar3& torque_j) = 0;
        
        #ifndef NVCC
        //! Get the name of the potential
        //! This function is pure virtual
        /*! \returns The potential name. Must be short and all lowercase, as this is the name energies will be logged as
            via analyze.log.
        */
        virtual static std::string getName() = 0;
        #endif

    protected:
	param_type params;
    };


#endif // __DirectionalEvaluatorPair__

