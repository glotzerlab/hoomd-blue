// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ManualVirtualParticleFiller.h
 * \brief Definition of class for manually backfilling solid boundaries with virtual particles.
 */

#ifndef MPCD_MANUAL_VIRTUAL_PARTICLE_FILLER_H_
#define MPCD_MANUAL_VIRTUAL_PARTICLE_FILLER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "VirtualParticleFiller.h"

#include <pybind11/pybind11.h>

#include <string>

namespace hoomd
    {
namespace mpcd
    {
//! Manually add virtual particles to the MPCD particle data
/*!
 * The ManualVirtualParticleFiller base class defines an interface for adding virtual particles
 * using a prescribed formula. The fill() method handles the basic tasks of appending a certain
 * number of virtual particles to the particle data. Each deriving class must then implement two
 * methods:
 *  1. computeNumFill(), which is the number of virtual particles to add.
 *  2. drawParticles(), which is the rule to determine where to put the particles.
 */
class PYBIND11_EXPORT ManualVirtualParticleFiller : public VirtualParticleFiller
    {
    public:
    ManualVirtualParticleFiller(std::shared_ptr<SystemDefinition> sysdef,
                                const std::string& type,
                                Scalar density,
                                std::shared_ptr<Variant> T);

    virtual ~ManualVirtualParticleFiller() { }

    //! Fill up virtual particles
    void fill(uint64_t timestep);

    protected:
    unsigned int m_N_fill;    //!< Number of particles to fill locally
    unsigned int m_first_tag; //!< First tag of locally held particles
    unsigned int m_first_idx; //!< Particle index to start adding from

    //! Compute the total number of particles to fill
    virtual void computeNumFill() { }

    //! Draw particles within the fill volume
    virtual void drawParticles(uint64_t timestep) { }
    };

namespace detail
    {
//! Export the ManualVirtualParticleFiller to python
void export_ManualVirtualParticleFiller(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_MANUAL_VIRTUAL_PARTICLE_FILLER_H_
