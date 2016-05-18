// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "MSDAnalyzer.h"

// include GPU classes
#ifdef ENABLE_CUDA
#endif

#include <boost/python.hpp>

using namespace boost::python;

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
BOOST_PYTHON_MODULE(_deprecated)
    {
    export_MSDAnalyzer();

#ifdef ENABLE_CUDA
#endif

    // boost 1.60.0 compatibility
    #if (BOOST_VERSION == 106000)
    register_ptr_to_python< boost::shared_ptr< MSDAnalyzer > >();

    #ifdef ENABLE_CUDA
    #endif

    #endif
    }
