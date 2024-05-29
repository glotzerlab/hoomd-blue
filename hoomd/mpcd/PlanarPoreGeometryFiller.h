// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/PlanarPoreGeometryFiller.h
 * \brief Definition of virtual particle filler for mpcd::PlanarPoreGeometry.
 */

#ifndef MPCD_SLIT_PORE_GEOMETRY_FILLER_H_
#define MPCD_SLIT_PORE_GEOMETRY_FILLER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ManualVirtualParticleFiller.h"
#include "PlanarPoreGeometry.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Adds virtual particles to the MPCD particle data for PlanarPoreGeometry
/*!
 * Particles are added to the volume that is overlapped by any of the cells that are also "inside"
 * the channel, subject to the grid shift.
 */
class PYBIND11_EXPORT PlanarPoreGeometryFiller : public mpcd::ManualVirtualParticleFiller
    {
    public:
    PlanarPoreGeometryFiller(std::shared_ptr<SystemDefinition> sysdef,
                             const std::string& type,
                             Scalar density,
                             std::shared_ptr<Variant> T,
                             std::shared_ptr<const mpcd::PlanarPoreGeometry> geom);

    virtual ~PlanarPoreGeometryFiller();

    std::shared_ptr<const mpcd::PlanarPoreGeometry> getGeometry() const
        {
        return m_geom;
        }

    void setGeometry(std::shared_ptr<const mpcd::PlanarPoreGeometry> geom)
        {
        m_geom = geom;
        notifyRecompute();
        }

    protected:
    std::shared_ptr<const mpcd::PlanarPoreGeometry> m_geom;

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
//! Export PlanarPoreGeometryFiller to python
void export_PlanarPoreGeometryFiller(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_SLIT_PORE_GEOMETRY_FILLER_H_
