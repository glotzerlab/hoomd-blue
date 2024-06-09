// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ParallelPlateGeometryFiller.h
 * \brief Definition of virtual particle filler for mpcd::ParallelPlateGeometry.
 */

#ifndef MPCD_PARALLEL_PLATE_GEOMETRY_FILLER_H_
#define MPCD_PARALLEL_PLATE_GEOMETRY_FILLER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ManualVirtualParticleFiller.h"
#include "ParallelPlateGeometry.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Adds virtual particles to the MPCD particle data for ParallelPlateGeometry
/*!
 * Particles are added to the volume that is overlapped by any of the cells that are also "inside"
 * the channel, subject to the grid shift.
 */
class PYBIND11_EXPORT ParallelPlateGeometryFiller : public mpcd::ManualVirtualParticleFiller
    {
    public:
    ParallelPlateGeometryFiller(std::shared_ptr<SystemDefinition> sysdef,
                                const std::string& type,
                                Scalar density,
                                std::shared_ptr<Variant> T,
                                std::shared_ptr<const mpcd::ParallelPlateGeometry> geom);

    virtual ~ParallelPlateGeometryFiller();

    virtual void fill(uint64_t timestep) override;

    std::shared_ptr<const mpcd::ParallelPlateGeometry> getGeometry() const
        {
        return m_geom;
        }

    void setGeometry(std::shared_ptr<const mpcd::ParallelPlateGeometry> geom)
        {
        m_geom = geom;
        }

    protected:
    std::shared_ptr<const mpcd::ParallelPlateGeometry> m_geom;
    Scalar m_y_min;      //!< Min y coordinate for filling
    Scalar m_y_max;      //!< Max y coordinate for filling
    unsigned int m_N_lo; //!< Number of particles to fill below channel
    unsigned int m_N_hi; //!< number of particles to fill above channel

    //! Compute the total number of particles to fill
    virtual void computeNumFill();

    //! Draw particles within the fill volume
    virtual void drawParticles(uint64_t timestep);
    };

namespace detail
    {
//! Export ParallelPlateGeometryFiller to python
void export_ParallelPlateGeometryFiller(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_PARALLEL_PLATE_GEOMETRY_FILLER_H_
