// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ActiveRotationalDiffusionUpdater.h
    \brief Declares an updater that actively diffuses particle orientations
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ForceCompute.h"
#include "hoomd/Integrator.h"
#include "hoomd/MeshDefinition.h"
#include "hoomd/Updater.h"
#include "hoomd/Variant.h"

#include <memory>
#include <pybind11/pybind11.h>

#pragma once

namespace hoomd
    {
namespace md
    {
/// Updates particle's orientations based on a given diffusion constant.
/** The updater accepts a variant rotational diffusion and updates the particle orientations of the
 * associated ActiveForceCompute's group (by calling m_active_force.rotationalDiffusion).
 *
 * Note: This was originally part of the ActiveForceCompute, and is separated to obey the idea that
 * force computes do not update the system directly, but updaters do. See GitHub issue (898). The
 * updater is just a shell that calls through to m_active_force due to the complexities of the logic
 * with the introduction of manifolds.
 *
 * If anyone has the time to do so, the implementation would be cleaner if moved to this updater.
 */
class PYBIND11_EXPORT MeshDynamicBondUpdater : public Updater
    {
    public:
    /// Constructor
    MeshDynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<Integrator> integrator,
                           std::shared_ptr<MeshDefinition> mesh,
                           Scalar T);

    /// Destructor
    virtual ~MeshDynamicBondUpdater();

    /// Update box interpolation based on provided timestep
    virtual void update(uint64_t timestep);

    Scalar getT()
        {
        return 1.0 / m_inv_T;
        };

    void setT(Scalar T)
        {
        m_inv_T = 1.0 / T;
        };

    private:
    std::shared_ptr<Integrator> m_integrator; //!< Integrator that advances time in this System
    std::shared_ptr<MeshDefinition> m_mesh;   //!< Active force to call rotationalDiffusion on
    Scalar m_inv_T;
    };

namespace detail
    {
/// Export the ActiveRotationalDiffusionUpdater to python
void export_MeshDynamicBondUpdater(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
