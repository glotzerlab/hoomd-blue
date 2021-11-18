// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: ajs42

#ifndef _COMPUTE_THERMO_HMA_TYPES_H_
#define _COMPUTE_THERMO_HMA_TYPES_H_

#include "hoomd/HOOMDMath.h"
/*! \file ComputeThermoHMATypes.h
    \brief Data structures common to both CPU and GPU implementations of ComputeThermoHMA
    */

namespace hoomd
    {
namespace md
    {
//! Enum for indexing the GPUArray of computed values
struct thermoHMA_index
    {
    //! The enum
    enum Enum
        {
        potential_energyHMA = 0, //!< Index for the potential energy in the GPUArray
        pressureHMA,             //!< Index for the potential energy in the GPUArray
        num_quantities           // final element to count number of quantities
        };
    };

    } // end namespace md
    } // end namespace hoomd
#endif
