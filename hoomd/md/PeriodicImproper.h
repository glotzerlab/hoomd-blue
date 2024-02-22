// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"

namespace hoomd
    {
namespace md
    {
struct periodic_improper_params
    {
    Scalar k;
    Scalar d;
    int n;
    Scalar chi_0;

#ifndef __HIPCC__
    periodic_improper_params() : k(0.), d(0.), n(0), chi_0(0.) { }

    periodic_improper_params(pybind11::dict v)
        : k(v["k"].cast<Scalar>()), d(v["d"].cast<Scalar>()), n(v["n"].cast<int>()),
          chi_0(v["chi0"].cast<Scalar>())
        {
        if (k <= 0)
            {
            throw std::runtime_error("Periodic improper K must be greater than 0.");
            }
        if (d != 1 && d != -1)
            {
            throw std::runtime_error("Periodic improper d must be -1 or 1.");
            }
        if (chi_0 < 0 || chi_0 >= Scalar(2 * M_PI))
            {
            throw std::runtime_error("Periodic improper chi_0 must be in the range [0, 2pi).");
            }
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["d"] = d;
        v["n"] = n;
        v["chi0"] = chi_0;
        return v;
        }
#endif
    } __attribute__((aligned(32)));

    } // namespace md
    } // namespace hoomd
