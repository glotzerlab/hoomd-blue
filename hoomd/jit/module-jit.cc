// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "PatchEnergyJIT.h"
#include "PatchEnergyJITUnion.h"

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
 create the hoomd python module and define the exports here.
 */

PYBIND11_MODULE(_jit, m)
    {
    export_PatchEnergyJIT(m);
    export_PatchEnergyJITUnion(m);
    export_ForceEnergyJIT(m);
    }
