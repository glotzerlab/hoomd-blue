// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// Maintainer: mspells

#include "hoomd/ForceCompute.h"
#include "hoomd/md/NeighborList.h"

#include <iterator>
#include <memory>
#include <pybind11/pybind11.h>

#include "DEMEvaluator.h"
#include "hoomd/GSDShapeSpecWriter.h"

/*! \file DEM2DForceCompute.h
  \brief Declares the DEM2DForceCompute class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __DEM2DFORCECOMPUTE_H__
#define __DEM2DFORCECOMPUTE_H__

namespace hoomd
    {
namespace dem
    {
//! Computes DEM 2D forces on each particle
/*! The total pair force is summed for each particle when compute() is called. Forces are only
  summed between neighboring particles with a separation distance less than \c r_cut. A NeighborList
  must be provided to identify these neighbors. Calling compute() in this class will in turn result
  in a call to the NeighborList's compute() to make sure that the neighbor list is up to date.

  Usage: Construct a DEM2DForceCompute, providing it an already constructed ParticleData and
  NeighborList. Then set parameters for each type by calling setParams.

  Forces can be computed directly by calling compute() and then retrieved with a call to acquire(),
  but a more typical usage will be to add the force compute to NVEUpdater or NVTUpdater.

  \ingroup computes
*/
template<typename Real, typename Real4, typename Potential>
class DEM2DForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    DEM2DForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<md::NeighborList> nlist,
                      Real r_cut,
                      Potential potential);

    //! Destructor
    virtual ~DEM2DForceCompute();

    //! Set the vertices for a particle
    virtual void setParams(unsigned int type, const pybind11::list& vertices);

    virtual void setRcut(Real r_cut)
        {
        m_r_cut = r_cut;
        }

    void connectDEMGSDShapeSpec(std::shared_ptr<GSDDumpWriter> writer);

    int slotWriteDEMGSDShapeSpec(gsd_handle& handle) const;

    std::string getTypeShape(const std::vector<vec2<Real>>& verts, const Real& radius) const;

    std::string encodeVertices(const std::vector<vec2<Real>>& verts) const;

    std::vector<std::string> getTypeShapeMapping(const std::vector<std::vector<vec2<Real>>>& verts,
                                                 const Real& radius) const;

    pybind11::list getTypeShapesPy();

#ifdef ENABLE_MPI
    //! Get requested ghost communication flags
    virtual CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        // by default, only request positions
        CommFlags flags(0);
        flags[comm_flag::orientation] = 1;
        flags[comm_flag::net_torque] = 1; // only used with rigid bodies

        flags |= ForceCompute::getRequestedCommFlags(timestep);
        return flags;
        }
#endif

    //! Returns true because we compute the torque
    virtual bool isAnisotropic()
        {
        return true;
        }

    protected:
    std::shared_ptr<md::NeighborList> m_nlist; //!< The neighborlist to use for the computation
    Real m_r_cut;                              //!< Cutoff radius beyond which the force is set to 0
    DEMEvaluator<Real, Real4, Potential>
        m_evaluator; //!< Object holding parameters and computation method for the potential
    std::vector<std::vector<vec2<Real>>> m_shapes; //!< Vertices for each type

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

    } // end namespace dem
    } // end namespace hoomd

#include "DEM2DForceCompute.cc"

#endif
