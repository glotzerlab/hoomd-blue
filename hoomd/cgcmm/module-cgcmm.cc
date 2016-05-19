// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "CGCMMAngleForceCompute.h"
#include "CGCMMForceCompute.h"

// include GPU classes
#ifdef ENABLE_CUDA
#include "CGCMMAngleForceComputeGPU.h"
#include "CGCMMForceComputeGPU.h"
#endif

#include <boost/python.hpp>

using namespace boost::python;

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
BOOST_PYTHON_MODULE(_cgcmm)
    {
    export_CGCMMAngleForceCompute();
    export_CGCMMForceCompute();

#ifdef ENABLE_CUDA
    export_CGCMMForceComputeGPU();
    export_CGCMMAngleForceComputeGPU();
#endif

    // boost 1.60.0 compatibility
    #if (BOOST_VERSION == 106000)

    #ifdef ENABLE_CUDA
    register_ptr_to_python< boost::shared_ptr< CGCMMAngleForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMForceComputeGPU > >();
    register_ptr_to_python< boost::shared_ptr< CGCMMAngleForceComputeGPU > >();

    #endif

    #endif
    }
