// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
struct area_conservation_param_t
    {
    Scalar k;
    Scalar A0;

#ifndef __HIPCC__
    area_conservation_param_t() : k(0), A0(0) { }

    area_conservation_param_t(pybind11::dict params)
        : k(params["k"].cast<Scalar>()), A0(params["A0"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["A0"] = A0;
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
