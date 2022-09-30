// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.




#ifndef __JanusFactor__
#define __JanusFactor__

#ifndef NVCC
#include <string>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
namespace py = pybind11;
#endif

#include <hoomd/HOOMDMath.h>

/*! \file ModulatorJanusSphere.h
    \brief This is a struct that handles Janus spheres of arbitrary balance
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

#ifdef SINGLE_PRECISION
#define _EXP(x) expf( (x) )
#else
#define _EXP(x) exp( (x) )
#endif


class JanusFactor
    {
    public:
        typedef Scalar2 param_type;

        DEVICE JanusFactor(const Scalar3& _dr,
                              const Scalar4& _qi,
                              const Scalar4& _qj,
                              const Scalar& _rcutsq,
                              const param_type& _params)
            : dr(_dr), qi(_qi), qj(_qj), params(_params)
            {
            // compute current janus direction vectors
            Scalar3 e = { 1,  0,  0 };
            quatrot(e,qi,ei);
            quatrot(e,qj,ej);

            // compute distance
            drsq = dr.x*dr.x+dr.y*dr.y+dr.z*dr.z;
            magdr = _SQRT(drsq);

            // compute dot products
            doti = -(dr.x*ei.x+dr.y*ei.y+dr.z*ei.z)/magdr;
            dotj =  (dr.x*ej.x+dr.y*ej.y+dr.z*ej.z)/magdr;
            }

        DEVICE inline Scalar Modulatori(void)
            {
            return Scalar(1.0)/(Scalar(1.0)+_EXP(-params.x*(doti-params.y)));
            }

        DEVICE inline Scalar Modulatorj(void)
            {
            return Scalar(1.0)/(Scalar(1.0)+_EXP(-params.x*(dotj-params.y)));
            }

        DEVICE Scalar ModulatorPrimei(void)
            {
            Scalar fact = Modulatori();
            return params.x*_EXP(-params.x*(doti-params.y))*fact*fact;
            }

        DEVICE Scalar ModulatorPrimej(void)
            {
            Scalar fact = Modulatorj();
            return params.x*_EXP(-params.x*(dotj-params.y))*fact*fact;
            }


        // things that get passed in to constructor
        Scalar3 dr;
        Scalar4 qi;
        Scalar4 qj;
        param_type params;
        // things that get calculated when constructor is called
        Scalar3 ei;
        Scalar3 ej;
        Scalar drsq;
        Scalar magdr;
        Scalar doti;
        Scalar dotj;
    };

#endif // __JanusFactor__

