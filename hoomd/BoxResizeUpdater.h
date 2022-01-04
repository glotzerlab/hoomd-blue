// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: joaander

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

#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#ifndef __BOXRESIZEUPDATER_H__
#define __BOXRESIZEUPDATER_H__

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
                     pybind11::object box1,
                     pybind11::object box2,
                     std::shared_ptr<Variant> variant,
                     std::shared_ptr<ParticleGroup> m_group);

    /// Destructor
    virtual ~BoxResizeUpdater();

    /// Gets particle scaling filter
    std::shared_ptr<ParticleGroup> getGroup()
        {
        return m_group;
        }

    /// Set a new initial box from a python object
    void setPyBox1(pybind11::object box1);

    /// Get the final box
    pybind11::object getPyBox1()
        {
        return m_py_box1;
        }

    /// Set a new final box from a python object
    void setPyBox2(pybind11::object box2);

    /// Get the final box
    pybind11::object getPyBox2()
        {
        return m_py_box2;
        }

    /// Set the variant for interpolation
    void setVariant(std::shared_ptr<Variant> variant)
        {
        m_variant = variant;
        }

    /// Get the variant for interpolation
    std::shared_ptr<Variant> getVariant()
        {
        return m_variant;
        }

    /// Get the current box for the given timestep
    BoxDim getCurrentBox(uint64_t timestep);

    /// Update box interpolation based on provided timestep
    virtual void update(uint64_t timestep);

    private:
    pybind11::object m_py_box1;             ///< The python box assoc with min
    pybind11::object m_py_box2;             ///< The python box assoc with max
    BoxDim& m_box1;                         ///< C++ box assoc with min
    BoxDim& m_box2;                         ///< C++ box assoc with max
    std::shared_ptr<Variant> m_variant;     //!< Variant that interpolates between boxes
    std::shared_ptr<ParticleGroup> m_group; //!< Selected particles to scale when resizing the box.
    };

namespace detail
    {
/// Export the BoxResizeUpdater to python
void export_BoxResizeUpdater(pybind11::module& m);

/// Get a BoxDim object from a pybind11::object or raise error
BoxDim& getBoxDimFromPyObject(pybind11::object box);

    } // end namespace detail
    } // end namespace hoomd
#endif
