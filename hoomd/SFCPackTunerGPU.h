// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SFCPackTunerGPU.h
    \brief Declares the SFCPackTunerGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_HIP
#include "Tuner.h"

#include "SFCPackTuner.h"
#include "SFCPackTunerGPU.cuh"
#include "GPUArray.h"

#include <memory>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <vector>
#include <utility>
#include <pybind11/pybind11.h>

#ifndef __SFCPACK_UPDATER_GPU_H__
#define __SFCPACK_UPDATER_GPU_H__

//! Sort the particles
/*! GPU implementation of SFCPackTuner

    \ingroup updaters
*/
class PYBIND11_EXPORT SFCPackTunerGPU : public SFCPackTuner
    {
    public:
        //! Constructor
        SFCPackTunerGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<Trigger> trigger);

        //! Destructor
        virtual ~SFCPackTunerGPU();

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
    };

//! Export the SFCPackTunerGPU class to python
void export_SFCPackTunerGPU(pybind11::module& m);

#endif // __SFC_PACK_UPDATER_GPU_H_


#endif // ENABLE_HIP
