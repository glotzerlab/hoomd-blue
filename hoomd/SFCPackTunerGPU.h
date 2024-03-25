// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SFCPackTunerGPU.h
    \brief Declares the SFCPackTunerGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_HIP
#include "Tuner.h"

#include "GPUArray.h"
#include "SFCPackTuner.h"
#include "SFCPackTunerGPU.cuh"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>
#include <pybind11/pybind11.h>
#include <utility>
#include <vector>

#ifndef __SFCPACK_UPDATER_GPU_H__
#define __SFCPACK_UPDATER_GPU_H__

namespace hoomd
    {
//! Sort the particles
/*! GPU implementation of SFCPackTuner

    \ingroup updaters
*/
class PYBIND11_EXPORT SFCPackTunerGPU : public SFCPackTuner
    {
    public:
    //! Constructor
    SFCPackTunerGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger);

    //! Destructor
    virtual ~SFCPackTunerGPU();

    protected:
    // reallocate internal data structure
    virtual void reallocate();

    private:
    GlobalArray<unsigned int> m_gpu_particle_bins; //!< Particle bins
    GlobalArray<unsigned int> m_gpu_sort_order;    //!< Generated sort order of the particles

    //! Helper function that actually performs the sort
    virtual void getSortedOrder2D();

    //! Helper function that actually performs the sort
    virtual void getSortedOrder3D();

    //! Apply the sorted order to the particle data
    virtual void applySortOrder();
    };

namespace detail
    {
//! Export the SFCPackTunerGPU class to python
void export_SFCPackTunerGPU(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif // __SFC_PACK_UPDATER_GPU_H_

#endif // ENABLE_HIP
