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
struct helfrich_param_t
    {
    Scalar k;
    Scalar H0;

#ifndef __HIPCC__
    helfrich_param_t() : k(0), H0(0) { }

    helfrich_param_t(pybind11::dict params)
        : k(params["k"].cast<Scalar>()), H0(params["H0"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["H0"] = H0;
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
