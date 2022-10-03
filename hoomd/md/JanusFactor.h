// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HOOMDMath.h"
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
namespace py = pybind11;

#ifndef __JanusFactor__
#define __JanusFactor__

#ifdef NVCC
#define DEVICE __device__
#else
#define DEVICE
#endif


namespace hoomd
    {
namespace md
    {
/*! \file JanusFactor.h
    \brief This is a struct that handles Janus spheres of arbitrary balance
    // TODO I don't think this is true anymore.
*/
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
            magdr = fast::sqrt(drsq);

            // compute dot products
            doti = -(dr.x*ei.x+dr.y*ei.y+dr.z*ei.z)/magdr;
            dotj =  (dr.x*ej.x+dr.y*ej.y+dr.z*ej.z)/magdr;
        }
    
    DEVICE inline Scalar Modulatori()
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.x*(doti-params.y)) );
        }
    
    DEVICE inline Scalar Modulatorj()
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.x*(dotj-params.y)) );
        }

    DEVICE Scalar ModulatorPrimei()
        {
            Scalar fact = Modulatori();
            return params.x * fast::exp(-params.x*(doti-params.y)) * fact * fact;
        }

    DEVICE Scalar ModulatorPrimej()
        {
            Scalar fact = Modulatorj();
            return params.x * fast::exp(-params.x*(dotj-params.y)) * fact * fact;
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

    } // end namespace md
    } // end namespace hoomd
