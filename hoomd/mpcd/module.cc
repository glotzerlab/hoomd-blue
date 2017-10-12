// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

// particle data
#include "ParticleData.h"
#include "ParticleDataSnapshot.h"
#include "Sorter.h"
#ifdef ENABLE_CUDA
#include "SorterGPU.h"
#endif // ENABLE_CUDA
#include "SystemData.h"
#include "SystemDataSnapshot.h"

// cell list
#include "CellList.h"
#include "CellThermoCompute.h"
#ifdef ENABLE_CUDA
#include "CellListGPU.h"
#include "CellThermoComputeGPU.h"
#endif // ENABLE_CUDA

// integration
#include "Integrator.h"

// Collision methods
#include "CollisionMethod.h"
#include "ATCollisionMethod.h"
#include "SRDCollisionMethod.h"
#ifdef ENABLE_CUDA
#include "ATCollisionMethodGPU.h"
#include "SRDCollisionMethodGPU.h"
#endif // ENABLE_CUDA

// Streaming methods
#include "StreamingGeometry.h"
#include "StreamingMethod.h"
#include "ConfinedStreamingMethod.h"
#ifdef ENABLE_CUDA
#include "ConfinedStreamingMethodGPU.h"
#endif // ENABLE_CUDA

// communicator
#ifdef ENABLE_MPI
#include "Communicator.h"
#ifdef ENABLE_CUDA
#include "CommunicatorGPU.h"
#endif // ENABLE_CUDA
#endif // ENABLE_MPI

#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

//! MPCD component
/*!
 * The mpcd namespace contains all classes, data members, and functions related
 * to performing multiparticle collision dynamics simulations.
 */
namespace mpcd
{

//! MPCD implementation details
/*!
 * The detail namespace contains classes and functions that are not part of the
 * MPCD public interface. These are not part of the public interface, and are
 * subject to change without notice.
 */
namespace detail
{
};

//! GPU functions for the MPCD component
/*!
 * The gpu namespace contains functions to drive CUDA kernels in the GPU
 * implementation. They are not part of the public interface for the MPCD component,
 * and are subject to change without notice.
 */
namespace gpu
{

//! GPU kernels for the MPCD component
/*!
 * The kernel namespace contains the kernels that do the work of a kernel driver
 * in the gpu namespace. They are not part of the public interface for the MPCD component,
 * and are subject to change without notice.
 */
namespace kernel
{
}; // end namespace kernel

}; // end namespace gpu

}; // end namespace mpcd

PYBIND11_MODULE(_mpcd, m)
    {
    mpcd::detail::export_ParticleData(m);
    mpcd::detail::export_ParticleDataSnapshot(m);
    mpcd::detail::export_Sorter(m);
    #ifdef ENABLE_CUDA
    mpcd::detail::export_SorterGPU(m);
    #endif // ENABLE_CUDA
    mpcd::detail::export_SystemData(m);
    mpcd::detail::export_SystemDataSnapshot(m);

    mpcd::detail::export_CellList(m);
    mpcd::detail::export_CellThermoCompute(m);
    #ifdef ENABLE_CUDA
    mpcd::detail::export_CellListGPU(m);
    mpcd::detail::export_CellThermoComputeGPU(m);
    #endif // ENABLE_CUDA

    mpcd::detail::export_Integrator(m);

    mpcd::detail::export_CollisionMethod(m);
    mpcd::detail::export_ATCollisionMethod(m);
    mpcd::detail::export_SRDCollisionMethod(m);
    #ifdef ENABLE_CUDA
    mpcd::detail::export_ATCollisionMethodGPU(m);
    mpcd::detail::export_SRDCollisionMethodGPU(m);
    #endif // ENABLE_CUDA

    mpcd::detail::export_boundary(m);
    mpcd::detail::export_BulkGeometry(m);
    mpcd::detail::export_SlitGeometry(m);

    mpcd::detail::export_StreamingMethod(m);
    mpcd::detail::export_ConfinedStreamingMethod<mpcd::detail::BulkGeometry>(m);
    mpcd::detail::export_ConfinedStreamingMethod<mpcd::detail::SlitGeometry>(m);
    #ifdef ENABLE_CUDA
    mpcd::detail::export_ConfinedStreamingMethodGPU<mpcd::detail::BulkGeometry>(m);
    mpcd::detail::export_ConfinedStreamingMethodGPU<mpcd::detail::SlitGeometry>(m);
    #endif // ENABLE_CUDA

    #ifdef ENABLE_MPI
    mpcd::detail::export_Communicator(m);
    #ifdef ENABLE_CUDA
    mpcd::detail::export_CommunicatorGPU(m);
    #endif // ENABLE_CUDA
    #endif // ENABLE_MPI
    }
