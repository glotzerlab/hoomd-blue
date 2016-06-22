// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "EAMForceCompute.h"

// include GPU classes
#ifdef ENABLE_CUDA
#include "EAMForceComputeGPU.h"
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

using namespace boost::python;

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
BOOST_PYTHON_MODULE(_cgcmm)
    {
    export_EAMForceCompute();

#ifdef ENABLE_CUDA
    export_EAMForceComputeGPU();
#endif

    // boost 1.60.0 compatibility
    #if (BOOST_VERSION == 106000)

    register_ptr_to_python< std::shared_ptr< EAMForceCompute > >();

    #ifdef ENABLE_CUDA
    register_ptr_to_python< std::shared_ptr< EAMForceComputeGPU > >();
    #endif

    #endif
    }
