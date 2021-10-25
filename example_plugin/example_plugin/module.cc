// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Include the defined classes that are to be exported to python
#include "ExampleUpdater.h"

#include <pybind11/pybind11.h>

using namespace hoomd::detail;

// specify the python module. Note that the name must explicitly match the PROJECT() name provided
// in CMakeLists (with an underscore in front)
PYBIND11_MODULE(_example_plugin, m)
    {
    export_ExampleUpdater(m);

#ifdef ENABLE_HIP
    export_ExampleUpdaterGPU(m);
#endif
    }
