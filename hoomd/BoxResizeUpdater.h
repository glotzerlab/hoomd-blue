// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxResizeUpdater.h
    \brief Declares an updater that resizes the simulation box of the system
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BoxDim.h"
#include "ParticleGroup.h"
#include "Updater.h"
#include "Variant.h"
#include "VectorVariant.h"

#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#pragma once

namespace hoomd
    {
/// Updates the simulation box over time
/** This simple updater gets the box lengths from specified variants and sets
 * those box sizes over time. As an option, particles can be rescaled with the
 * box lengths or left where they are. Note: rescaling particles does not work
 * properly in MPI simulations.
 * \ingroup updaters
 */
class PYBIND11_EXPORT BoxResizeUpdater : public Updater
    {
    public:
    /// Constructor
    BoxResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<Trigger> trigger,
                     std::shared_ptr<VectorVariantBox> box,
                     std::shared_ptr<ParticleGroup> m_group);

    /// Destructor
    virtual ~BoxResizeUpdater();

    /// Set the box variant
    void setBox(std::shared_ptr<VectorVariantBox> box)
        {
        m_box = box;
        }

    /// Get the box variant
    std::shared_ptr<VectorVariantBox> getBox()
        {
        return m_box;
        }

    /// Gets particle scaling filter
    std::shared_ptr<ParticleGroup> getGroup()
        {
        return m_group;
        }

    /// Get the current box for the given timestep
    BoxDim getCurrentBox(uint64_t timestep);

    /// Update box interpolation based on provided timestep
    virtual void update(uint64_t timestep);

    /// Scale particles to the new box and wrap any back into the box
    virtual void scaleAndWrapParticles(const BoxDim& cur_box, const BoxDim& new_box);

    protected:
    /// Box as a function of time.
    std::shared_ptr<VectorVariantBox> m_box;

    /// Selected particles to scale when resizing the box.
    std::shared_ptr<ParticleGroup> m_group;
    };

namespace detail
    {
/// Export the BoxResizeUpdater to python
void export_BoxResizeUpdater(pybind11::module& m);
    } // end namespace detail
    } // end namespace hoomd
