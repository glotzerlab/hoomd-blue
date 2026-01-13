// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __BOX_DEFORMER_H__
#define __BOX_DEFORMER_H__

#include "hoomd/BoxDim.h"
#include "hoomd/ParticleData.h"
#include "hoomd/SystemDefinition.h"

#include <memory>
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
/** Perform box deformation.
    Implement methods for simulation under box deformation.
*/
class PYBIND11_EXPORT BoxDeformer
    {
    public:
    /// Constructor
    BoxDeformer(std::shared_ptr<SystemDefinition> sysdef);

    /// Destructor
    virtual ~BoxDeformer();

    /// Set the time step size to be consistent with the Integrator's
    virtual void setDeltaT(Scalar deltaT);

    // Apply deformation
    void update(uint64_t timestep);

    protected:
    std::shared_ptr<SystemDefinition>
        m_sysdef;                          //!< The system definition this method is associated with
    std::shared_ptr<ParticleData> m_pdata; //!< The particle data this method is associated with
    std::shared_ptr<const ExecutionConfiguration>
        m_exec_conf; //!< Stored shared ptr to the execution configuration
    Scalar m_deltaT; //!< The time step

    virtual BoxDim computeNewBox(uint64_t timestep, const BoxDim& old_box);

    virtual void processAfterDeformation(const BoxDim& old_box, BoxDim& new_box);
    };

namespace detail
    {
/// Export the BoxDeformer class to python
void export_BoxDeformer(pybind11::module& m);
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // #ifndef __BOX_DEFORMER_H__
