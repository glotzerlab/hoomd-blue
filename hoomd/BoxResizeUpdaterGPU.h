// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxResizeUpdater.h
    \brief Declares an updater that resizes the simulation box of the system
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BoxResizeUpdater.h"

#ifndef __BOX_RESIZE_UPDATER_GPU_H__
#define __BOX_RESIZE_UPDATER_GPU_H__

namespace hoomd
    {
/// Updates the simulation box over time using the GPU
/** This simple updater gets the box lengths from specified variants and sets
 * those box sizes over time. As an option, particles can be rescaled with the
 * box lengths or left where they are. Note: rescaling particles does not work
 * properly in MPI simulations with HPMC.
 * \ingroup updaters
 */
class PYBIND11_EXPORT BoxResizeUpdaterGPU : public BoxResizeUpdater
    {
    public:
    /// Constructor
    BoxResizeUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<Trigger> trigger,
                        std::shared_ptr<VectorVariantBox> box,
                        std::shared_ptr<ParticleGroup> m_group);

    /// Destructor
    virtual ~BoxResizeUpdaterGPU();

    /// Scale particles to the new box and wrap any others back into the box
    virtual void scaleAndWrapParticles(const BoxDim& cur_box, const BoxDim& new_box);

    private:
    /// Autotuner for block size (scale kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_scale;
    /// Autotuner for block size (wrap kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_wrap;
    };

namespace detail
    {
/// Export the BoxResizeUpdaterGPU to python
void export_BoxResizeUpdaterGPU(pybind11::module& m);
    } // end namespace detail
    } // end namespace hoomd
#endif // __BOX_RESIZE_UPDATER_GPU_H__
