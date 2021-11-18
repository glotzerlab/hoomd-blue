// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "EAMForceCompute.h"

// include GPU classes
#ifdef ENABLE_HIP
#include "EAMForceComputeGPU.h"
#endif

#include <pybind11/pybind11.h>

using namespace hoomd::metal::detail;

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
 create the hoomd python module and define the exports here.
 */
PYBIND11_MODULE(_metal, m)
    {
    export_EAMForceCompute(m);

#ifdef ENABLE_HIP
    export_EAMForceComputeGPU(m);
#endif
    }
