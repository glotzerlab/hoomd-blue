// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/SlitGeometryFiller.h
 * \brief Definition of virtual particle filler for mpcd::detail::SlitGeometry.
 */

#ifndef MPCD_SLIT_GEOMETRY_FILLER_H_
#define MPCD_SLIT_GEOMETRY_FILLER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "VirtualParticleFiller.h"
#include "SlitGeometry.h"

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Adds virtual particles to the MPCD particle data for SlitGeometry
/*!
 * Particles are added to the volume that is overlapped by any of the cells that are also "inside" the channel,
 * subject to the grid shift.
 */
class PYBIND11_EXPORT SlitGeometryFiller : public mpcd::VirtualParticleFiller
    {
    public:
        SlitGeometryFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                           Scalar density,
                           unsigned int type,
                           std::shared_ptr<::Variant> T,
                           unsigned int seed,
                           std::shared_ptr<const mpcd::detail::SlitGeometry> geom);

        virtual ~SlitGeometryFiller();

        void setGeometry(std::shared_ptr<const mpcd::detail::SlitGeometry> geom)
            {
            m_geom = geom;
            }

    protected:
        std::shared_ptr<const mpcd::detail::SlitGeometry> m_geom;
        Scalar m_z_min; //!< Min z coordinate for filling
        Scalar m_z_max; //!< Max z coordinate for filling
        unsigned int m_N_lo;    //!< Number of particles to fill below channel
        unsigned int m_N_hi;    //!< number of particles to fill above channel

        //! Compute the total number of particles to fill
        virtual void computeNumFill();

        //! Draw particles within the fill volume
        virtual void drawParticles(unsigned int timestep);
    };

namespace detail
{
//! Export SlitGeometryFiller to python
void export_SlitGeometryFiller(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_SLIT_GEOMETRY_FILLER_H_
