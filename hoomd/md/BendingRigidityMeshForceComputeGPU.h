// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BendingRigidityMeshForceCompute.h"
#include "BendingRigidityMeshForceComputeGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file BendingRigidityMeshForceComputeGPU.h
    \brief Declares a class for computing bending rigidity energy forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __BENDINGRIGIDITYMESHFORCECOMPUTE_GPU_H__
#define __BENDINGRIGIDITYMESHFORCECOMPUTE_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes bending rigidity forces on the mesh on the GPU
/*! Bending rigidity forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT BendingRigidityMeshForceComputeGPU : public BendingRigidityMeshForceCompute
    {
    public:
    //! Constructs the compute
    BendingRigidityMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<MeshDefinition> meshdef);
    //! Destructor
    ~BendingRigidityMeshForceComputeGPU();

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the BendingRigidityMeshForceComputeGPU class to python
void export_BendingRigidityMeshForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
