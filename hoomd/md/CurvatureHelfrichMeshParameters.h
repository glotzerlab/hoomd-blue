// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/HOOMDMath.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

#pragma once

namespace hoomd
    {
namespace md
    {

struct curvature_helfrich_param_t
    {
    Scalar k;
    Scalar C0;
    Scalar eps_kT;
    unsigned int tag_max;

#ifndef __HIPCC__
    curvature_helfrich_param_t() : k(0), C0(0), eps_kT(0), tag_max(0) { }

    curvature_helfrich_param_t(pybind11::dict params)
        : k(params["k"].cast<Scalar>()), C0(params["C0"].cast<Scalar>()), eps_kT(params["eps_kT"].cast<Scalar>()),tag_max(params["tag_max"].cast<unsigned int>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["C0"] = C0;
        v["eps_kT"] = eps_kT;
        v["tag_max"] = tag_max;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(4)));
#else
    __attribute__((aligned(8)));
#endif

    } // namespace md
    } // namespace hoomd
