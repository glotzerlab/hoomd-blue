// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "MSDAnalyzer.h"
#include "HOOMDDumpWriter.h"
#include "POSDumpWriter.h"
#include "HOOMDInitializer.h"
#include "RandomGenerator.h"

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
    export_HOOMDDumpWriter();
    export_POSDumpWriter();
    export_HOOMDInitializer();
    export_RandomGenerator();

#ifdef ENABLE_CUDA
#endif

    // boost 1.60.0 compatibility
    #if (BOOST_VERSION == 106000)
    register_ptr_to_python< std::shared_ptr< MSDAnalyzer > >();
    register_ptr_to_python< std::shared_ptr< HOOMDDumpWriter > >();
    register_ptr_to_python< std::shared_ptr< POSDumpWriter > >();
    register_ptr_to_python< std::shared_ptr< HOOMDInitializer > >();
    register_ptr_to_python< std::shared_ptr< RandomGenerator > >();
    register_ptr_to_python< std::shared_ptr< PolymerParticleGenerator > >();

    #ifdef ENABLE_CUDA
    #endif

    #endif
    }
