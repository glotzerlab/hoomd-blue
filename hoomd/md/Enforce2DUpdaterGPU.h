// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file Enforce2DUpdaterGPU.h
    \brief Declares the Enforce2DUpdaterGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Enforce2DUpdater.h"

#include <memory>

#ifndef __ENFORCE2DUPDATER_GPU_H__
#define __ENFORCE2DUPDATER_GPU_H__

//! NVE via velocity verlet on the GPU
/*! Enforce2DUpdaterGPU implements exactly the same calculations as NVEUpdater, but on the GPU.

    The GPU kernel that accomplishes this can be found in gpu_nve_kernel.cu

    \ingroup updaters
*/
class PYBIND11_EXPORT Enforce2DUpdaterGPU : public Enforce2DUpdater
    {
    public:
        //! Constructor
        Enforce2DUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    };

//! Exports the Enforce2DUpdaterGPU class to python
void export_Enforce2DUpdaterGPU(pybind11::module& m);

#endif
