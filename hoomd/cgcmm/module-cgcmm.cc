// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "CGCMMAngleForceCompute.h"
#include "CGCMMForceCompute.h"

// include GPU classes
#ifdef ENABLE_CUDA
#include "CGCMMAngleForceComputeGPU.h"
#include "CGCMMForceComputeGPU.h"
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
PYBIND11_MODULE(_cgcmm, m)
    {
    export_CGCMMAngleForceCompute(m);
    export_CGCMMForceCompute(m);

#ifdef ENABLE_CUDA
    export_CGCMMForceComputeGPU(m);
    export_CGCMMAngleForceComputeGPU(m);
#endif
    }
