// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/SlitPoreGeometryFiller.h
 * \brief Definition of virtual particle filler for mpcd::detail::SlitPoreGeometry.
 */

#ifndef MPCD_SLIT_PORE_GEOMETRY_FILLER_H_
#define MPCD_SLIT_PORE_GEOMETRY_FILLER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "SlitPoreGeometry.h"
#include "VirtualParticleFiller.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Adds virtual particles to the MPCD particle data for SlitPoreGeometry
/*!
 * Particles are added to the volume that is overlapped by any of the cells that are also "inside"
 * the channel, subject to the grid shift.
 */
class PYBIND11_EXPORT SlitPoreGeometryFiller : public mpcd::VirtualParticleFiller
    {
    public:
    SlitPoreGeometryFiller(std::shared_ptr<SystemDefinition> sysdef,
                           Scalar density,
                           unsigned int type,
                           std::shared_ptr<Variant> T,
                           uint16_t seed,
                           std::shared_ptr<const mpcd::detail::SlitPoreGeometry> geom);

    virtual ~SlitPoreGeometryFiller();

    void setGeometry(std::shared_ptr<const mpcd::detail::SlitPoreGeometry> geom)
        {
        m_geom = geom;
        notifyRecompute();
        }

    protected:
    std::shared_ptr<const mpcd::detail::SlitPoreGeometry> m_geom;

    const static unsigned int MAX_BOXES = 6; //!< Maximum number of boxes to fill
    unsigned int m_num_boxes;                //!< Number of boxes to use in filling
    GPUArray<Scalar4> m_boxes;               //!< Boxes to use in filling
    GPUArray<uint2> m_ranges;                //!< Particle tag ranges for filling

    //! Compute the total number of particles to fill
    virtual void computeNumFill();

    //! Draw particles within the fill volume
    virtual void drawParticles(uint64_t timestep);

    private:
    bool m_needs_recompute;
    Scalar3 m_recompute_cache;
    void notifyRecompute()
        {
        m_needs_recompute = true;
        }
    };

namespace detail
    {
//! Export SlitPoreGeometryFiller to python
void export_SlitPoreGeometryFiller(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_SLIT_PORE_GEOMETRY_FILLER_H_
