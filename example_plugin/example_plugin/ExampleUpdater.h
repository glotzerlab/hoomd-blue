// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// **********************
// This is a simple example code written for no function purpose other then to demonstrate the steps needed to write a
// c++ source code plugin for HOOMD-Blue. This example includes an example Updater class, but it can just as easily be
// replaced with a ForceCompute, Integrator, or any other C++ code at all.

// inclusion guard
#ifndef _EXAMPLE_UPDATER_H_
#define _EXAMPLE_UPDATER_H_

/*! \file ExampleUpdater.h
    \brief Declaration of ExampleUpdater
*/

#include <hoomd/Updater.h>

// pybind11 is used to create the python bindings to the C++ object,
// but not if we are compiling GPU kernels
#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

// (if you really don't want to include the whole hoomd.h, you can include individual files IF AND ONLY IF
// hoomd_config.h is included first)
// For example:
// #include <hoomd/Updater.h>

// Second, we need to declare the class. One could just as easily use any class in HOOMD as a template here, there are
// no restrictions on what a template can do

//! A nonsense particle updater written to demonstrate how to write a plugin
/*! This updater simply sets all of the particle's velocities to 0 when update() is called.
*/
class ExampleUpdater : public Updater
    {
    public:
        //! Constructor
        ExampleUpdater(std::shared_ptr<SystemDefinition> sysdef);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);
    };

//! Export the ExampleUpdater class to python
void export_ExampleUpdater(pybind11::module& m);

// Third, this class offers a GPU accelerated method in order to demonstrate how to include CUDA code in pluins
// we need to declare a separate class for that (but only if ENABLE_CUDA is set)

#ifdef ENABLE_CUDA

//! A GPU accelerated nonsense particle updater written to demonstrate how to write a plugin w/ CUDA code
/*! This updater simply sets all of the particle's velocities to 0 (on the GPU) when update() is called.
*/
class ExampleUpdaterGPU : public ExampleUpdater
    {
    public:
        //! Constructor
        ExampleUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);
    };

//! Export the ExampleUpdaterGPU class to python
void export_ExampleUpdaterGPU(pybind11::module& m);

#endif // ENABLE_CUDA

#endif // _EXAMPLE_UPDATER_H_
