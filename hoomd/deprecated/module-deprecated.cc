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

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
PYBIND11_PLUGIN(_deprecated)
	{
	pybind11::module m("_deprecated");

    export_MSDAnalyzer(m);
    export_HOOMDDumpWriter(m);
    export_POSDumpWriter(m);
    export_HOOMDInitializer(m);
    export_RandomGenerator(m);

    return m.ptr();

#ifdef ENABLE_CUDA
#endif

    // // boost 1.60.0 compatibility
    // #if (BOOST_VERSION == 106000)
    // register_ptr_to_python< std::shared_ptr< MSDAnalyzer > >();
    // register_ptr_to_python< std::shared_ptr< HOOMDDumpWriter > >();
    // register_ptr_to_python< std::shared_ptr< POSDumpWriter > >();
    // register_ptr_to_python< std::shared_ptr< HOOMDInitializer > >();
    // register_ptr_to_python< std::shared_ptr< RandomGenerator > >();
    // register_ptr_to_python< std::shared_ptr< PolymerParticleGenerator > >();
    //
    // #ifdef ENABLE_CUDA
    // #endif
    //
    // #endif
    }
