// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SphereResizeUpdater.h
    \brief Declares an updater that resizes the simulation sphere of the system
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "Sphere.h"
#include "ParticleGroup.h"
#include "Updater.h"
#include "Variant.h"

#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>

#ifndef __SPHERERESIZEUPDATER_H__
#define __SPHERERESIZEUPDATER_H__

namespace hoomd
    {
/// Updates the simulation sphere over time
/** This simple updater gets the sphere radius from specified variants and sets
 * those radii over time. As an option, particles can be rescaled with the
 * sphere radius or left where they are. Note: rescaling particles does not work
 * properly in MPI simulations.
 * \ingroup updaters
 */
class PYBIND11_EXPORT SphereResizeUpdater : public Updater
    {
    public:
    /// Constructor
    SphereResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<Trigger> trigger,
                     std::shared_ptr<Sphere> sphere1,
                     std::shared_ptr<Sphere> sphere2,
                     std::shared_ptr<Variant> variant,
                     std::shared_ptr<ParticleGroup> m_group);

    /// Destructor
    virtual ~SphereResizeUpdater();

    /// Get the current m_sphere1
    std::shared_ptr<Sphere> getSphere1();

    /// Set a new m_sphere_1
    void setSphere1(std::shared_ptr<Sphere> sphere1);

    /// Get the current m_sphere2
    std::shared_ptr<Sphere> getSphere2();

    /// Set a new m_sphere_2
    void setSphere2(std::shared_ptr<Sphere> sphere2);

    /// Gets particle scaling filter
    std::shared_ptr<ParticleGroup> getGroup()
        {
        return m_group;
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

    /// Get the current sphere for the given timestep
    Sphere getCurrentSphere(uint64_t timestep);

    /// Update sphere interpolation based on provided timestep
    virtual void update(uint64_t timestep);

    /// Scale particles to the new sphere and wrap any back into the sphere
    /// Nope it's a closed system (Gabby)
    //virtual void scaleAndWrapParticles(const Sphere& cur_sphere, const Sphere& new_sphere);

    protected:
    std::shared_ptr<Sphere> m_sphere1;         ///< C++ sphere assoc with min
    std::shared_ptr<Sphere> m_sphere2;         ///< C++ sphere assoc with max
    std::shared_ptr<Variant> m_variant;     //!< Variant that interpolates between spheres 
    std::shared_ptr<ParticleGroup> m_group; //!< Selected particles to scale when resizing the sphere.
    };

namespace detail
    {
/// Export the SphereResizeUpdater to python
void export_SphereResizeUpdater(pybind11::module& m);
    } // end namespace detail
    } // end namespace hoomd
#endif
