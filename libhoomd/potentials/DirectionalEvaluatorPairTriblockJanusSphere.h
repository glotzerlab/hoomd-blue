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

#ifndef __DirectionalEvaluatorPairTriblockJanusSphere__
#define __DirectionalEvaluatorPairTriblockJanusSphere__

#ifndef NVCC
#include <string>
#endif

#include "HOOMDMath.h"

/*! \file DirectionalEvaluatorPairTriblockJanusSphere.h
    \brief Defines an evaluator struct for Triblock Janus Spheres
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

struct EvaluatorPairTriblockJanusSphereStruct
    {
    EvaluatorPairTriblockJanusSphereStruct(Scalar3& _dr, Scalar4& _qi,
		    Scalar4& _qj, Scalar2 _params) :
        dr(_dr), qi(_qi), qj(_qj), params(_params)
        {
	// compute current janus direction vectors
        Scalar3 e = { 0 , 0 , 1 };
	quatrot(e,qi,ei);
	quatrot(e,qj,ej);

	// compute distance
	drsq = dr.x*dr.x+dr.y*dr.y+dr.z*dr.z;
	magdr = sqrt(drsq);

	// compute dot products
	doti =  (dr.x*ei.x+dr.y*ei.y+dr.z*ei.z)/magdr;
	dotj = -(dr.x*ej.x+dr.y*ej.y+dr.z*ej.z)/magdr;
	}

    DEVICE inline Scalar Modulatori(void)
        {
	return Scalar(1.0)/(1.0+exp(-params.x*(doti-params.y)))+
	       Scalar(1.0)/(1.0+exp(params.x*(doti+params.y)));
	}

    DEVICE inline Scalar Modulatorj(void)
        {
	return Scalar(1.0)/(1.0+exp(-params.x*(dotj-params.y)))+
	       Scalar(1.0)/(1.0+exp(params.x*(dotj+params.y)));
	}

    DEVICE Scalar ModulatorPrimei(void)
        {
	Scalar fact1 = Scalar(1.0)/(1.0+exp(-params.x*(doti-params.y)));
	Scalar fact2 = Scalar(1.0)/(1.0+exp(params.x*(doti+params.y)));
	return params.x*(exp(-params.x*(doti-params.y))*fact1*fact1-exp(params.x*(doti+params.y))*fact2*fact2);
	}

    DEVICE Scalar ModulatorPrimej(void)
        {
	Scalar fact1 = Scalar(1.0)/(1.0+exp(-params.x*(dotj-params.y)));
	Scalar fact2 = Scalar(1.0)/(1.0+exp(params.x*(dotj+params.y)));
	return params.x*(exp(-params.x*(dotj-params.y))*fact1*fact1-exp(params.x*(dotj+params.y))*fact2*fact2);
	}

    // things that get passed in to constructor
    Scalar3 dr;
    Scalar4 qi;
    Scalar4 qj;
    Scalar2 params;
    // things that get calculated when constructor is called
    Scalar3 ei;
    Scalar3 ej;
    Scalar drsq;
    Scalar magdr;
    Scalar doti;
    Scalar dotj;
    };

#endif // __DirectionalEvaluatorPairTriblockJanusSphere__

