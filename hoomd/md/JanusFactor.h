// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.


#ifndef __JANUS_FACTOR_H__
#define __JANUS_FACTOR_H__

#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"

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
*/
class JanusFactor
{
public:

    struct param_type
    {
        param_type()
            {
            }

        param_type(pybind11::dict params)
            : cosalpha( fast::cos(params["alpha"].cast<Scalar>()) ), // TODO implement bound checking
              omega(params["omega"].cast<Scalar>())
            {
            }

        pybind11::dict asDict()
            {
                pybind11::dict v;

                v["alpha"] = fast::acos(cosalpha);
                v["omega"] = omega;

                return v;
            }

        Scalar cosalpha;
        Scalar omega;
    }
#ifdef SINGLE_PRECISION
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    DEVICE JanusFactor(const Scalar3& _dr,
                       const Scalar4& _qi,
                       const Scalar4& _qj,
                       const Scalar& _rcutsq,
                       const param_type& _params)
        : dr(_dr), qi(_qi), qj(_qj), params(_params)
        {
            // compute current janus direction vectors
            vec3<Scalar> e { make_scalar3(1, 0, 0) };
            vec3<Scalar> ei;
            vec3<Scalar> ej;

            ei = rotate(quat<Scalar>(qi), e);
            ej = rotate(quat<Scalar>(qj), e);

            // compute distance
            drsq = dot(dr, dr);
            magdr = fast::sqrt(drsq);

            // compute dot products
            doti = dot(dr, ei) / magdr;
            // doti = -(dr.x*ei.x+dr.y*ei.y+dr.z*ei.z)/magdr; // TODO why was this negative?
            dotj = dot(dr, ej) / magdr;
            // dotj =  (dr.x*ej.x+dr.y*ej.y+dr.z*ej.z)/magdr;
        }

    DEVICE inline Scalar Modulatori()
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.omega*(doti-params.cosalpha)) );
        }

    DEVICE inline Scalar Modulatorj()
        {
            return Scalar(1.0) / ( Scalar(1.0) + fast::exp(-params.omega*(dotj-params.cosalpha)) );
        }

    DEVICE Scalar ModulatorPrimei()
        {
            Scalar fact = Modulatori();
            return params.omega * fast::exp(-params.omega*(doti-params.cosalpha)) * fact * fact;
        }

    DEVICE Scalar ModulatorPrimej()
        {
            Scalar fact = Modulatorj();
            return params.omega * fast::exp(-params.omega*(dotj-params.cosalpha)) * fact * fact;
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



    } // end namespace md
    } // end namespace hoomd

#endif // __JANUS_FACTOR_H__
