// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SFCPackUpdaterGPU.h
    \brief Declares the SFCPackUpdaterGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_CUDA

#include "Updater.h"

#include "SFCPackUpdater.h"
#include "SFCPackUpdaterGPU.cuh"
#include "GPUArray.h"

#include <memory>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <vector>
#include <utility>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __SFCPACK_UPDATER_GPU_H__
#define __SFCPACK_UPDATER_GPU_H__

//! Sort the particles
/*! GPU implementation of SFCPackUpdater

    \ingroup updaters
*/
class PYBIND11_EXPORT SFCPackUpdaterGPU : public SFCPackUpdater
    {
    public:
        //! Constructor
        SFCPackUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~SFCPackUpdaterGPU();

    protected:
        // reallocate internal data structure
        virtual void reallocate();

    private:
        GlobalArray<unsigned int> m_gpu_particle_bins;    //!< Particle bins
        GlobalArray<unsigned int> m_gpu_sort_order;       //!< Generated sort order of the particles

        //! Helper function that actually performs the sort
        virtual void getSortedOrder2D();

        //! Helper function that actually performs the sort
        virtual void getSortedOrder3D();

        //! Apply the sorted order to the particle data
        virtual void applySortOrder();

        mgpu::ContextPtr m_mgpu_context;                    //!< MGPU context (for sorting)
    };

//! Export the SFCPackUpdaterGPU class to python
void export_SFCPackUpdaterGPU(pybind11::module& m);

#endif // __SFC_PACK_UPDATER_GPU_H_


#endif // ENABLE_CUDA
