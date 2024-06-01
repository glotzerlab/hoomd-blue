// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

// particle data
#include "Sorter.h"
#ifdef ENABLE_HIP
#include "SorterGPU.h"
#endif // ENABLE_HIP

// cell list
#include "CellList.h"
#include "CellThermoCompute.h"
#ifdef ENABLE_HIP
#include "CellListGPU.h"
#include "CellThermoComputeGPU.h"
#endif // ENABLE_HIP

// forces
#include "BlockForce.h"
#include "ConstantForce.h"
#include "NoForce.h"
#include "SineForce.h"

// integration
#include "Integrator.h"

// Collision methods
#include "ATCollisionMethod.h"
#include "CollisionMethod.h"
#include "SRDCollisionMethod.h"
#ifdef ENABLE_HIP
#include "ATCollisionMethodGPU.h"
#include "SRDCollisionMethodGPU.h"
#endif // ENABLE_HIP

// Streaming methods
#include "BounceBackStreamingMethod.h"
#include "BulkStreamingMethod.h"
#include "StreamingGeometry.h"
#include "StreamingMethod.h"
#ifdef ENABLE_HIP
#include "BounceBackStreamingMethodGPU.h"
#include "BulkStreamingMethodGPU.h"
#endif // ENABLE_HIP

// integration methods
#include "BounceBackNVE.h"
#ifdef ENABLE_HIP
#include "BounceBackNVEGPU.h"
#endif

// virtual particle fillers
#include "ManualVirtualParticleFiller.h"
#include "ParallelPlateGeometryFiller.h"
#include "PlanarPoreGeometryFiller.h"
#include "VirtualParticleFiller.h"
#ifdef ENABLE_HIP
#include "ParallelPlateGeometryFillerGPU.h"
#include "PlanarPoreGeometryFillerGPU.h"
#endif // ENABLE_HIP

// communicator
#ifdef ENABLE_MPI
#include "Communicator.h"
#ifdef ENABLE_HIP
#include "CommunicatorGPU.h"
#endif // ENABLE_HIP
#endif // ENABLE_MPI

#include <pybind11/pybind11.h>

namespace hoomd
    {
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
    } // end namespace kernel

    } // end namespace gpu

    } // end namespace mpcd
    } // end namespace hoomd

using namespace hoomd;

PYBIND11_MODULE(_mpcd, m)
    {
    mpcd::detail::export_Sorter(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_SorterGPU(m);
#endif // ENABLE_HIP

    mpcd::detail::export_CellList(m);
    mpcd::detail::export_CellThermoCompute(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_CellListGPU(m);
    mpcd::detail::export_CellThermoComputeGPU(m);
#endif // ENABLE_HIP

    mpcd::detail::export_BlockForce(m);
    mpcd::detail::export_ConstantForce(m);
    mpcd::detail::export_NoForce(m);
    mpcd::detail::export_SineForce(m);

    mpcd::detail::export_Integrator(m);

    mpcd::detail::export_CollisionMethod(m);
    mpcd::detail::export_ATCollisionMethod(m);
    mpcd::detail::export_SRDCollisionMethod(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_ATCollisionMethodGPU(m);
    mpcd::detail::export_SRDCollisionMethodGPU(m);
#endif // ENABLE_HIP

    mpcd::detail::export_ParallelPlateGeometry(m);
    mpcd::detail::export_PlanarPoreGeometry(m);

    mpcd::detail::export_StreamingMethod(m);
    // bulk
    mpcd::detail::export_BulkStreamingMethod<mpcd::BlockForce>(m);
    mpcd::detail::export_BulkStreamingMethod<mpcd::ConstantForce>(m);
    mpcd::detail::export_BulkStreamingMethod<mpcd::NoForce>(m);
    mpcd::detail::export_BulkStreamingMethod<mpcd::SineForce>(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_BulkStreamingMethodGPU<mpcd::BlockForce>(m);
    mpcd::detail::export_BulkStreamingMethodGPU<mpcd::ConstantForce>(m);
    mpcd::detail::export_BulkStreamingMethodGPU<mpcd::NoForce>(m);
    mpcd::detail::export_BulkStreamingMethodGPU<mpcd::SineForce>(m);
#endif // ENABLE_HIP
    // parallel plate
    mpcd::detail::export_BounceBackStreamingMethod<mpcd::ParallelPlateGeometry, mpcd::BlockForce>(
        m);
    mpcd::detail::export_BounceBackStreamingMethod<mpcd::ParallelPlateGeometry,
                                                   mpcd::ConstantForce>(m);
    mpcd::detail::export_BounceBackStreamingMethod<mpcd::ParallelPlateGeometry, mpcd::NoForce>(m);
    mpcd::detail::export_BounceBackStreamingMethod<mpcd::ParallelPlateGeometry, mpcd::SineForce>(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_BounceBackStreamingMethodGPU<mpcd::ParallelPlateGeometry,
                                                      mpcd::BlockForce>(m);
    mpcd::detail::export_BounceBackStreamingMethodGPU<mpcd::ParallelPlateGeometry,
                                                      mpcd::ConstantForce>(m);
    mpcd::detail::export_BounceBackStreamingMethodGPU<mpcd::ParallelPlateGeometry, mpcd::NoForce>(
        m);
    mpcd::detail::export_BounceBackStreamingMethodGPU<mpcd::ParallelPlateGeometry, mpcd::SineForce>(
        m);
#endif // ENABLE_HIP
    // planar pore
    mpcd::detail::export_BounceBackStreamingMethod<mpcd::PlanarPoreGeometry, mpcd::BlockForce>(m);
    mpcd::detail::export_BounceBackStreamingMethod<mpcd::PlanarPoreGeometry, mpcd::ConstantForce>(
        m);
    mpcd::detail::export_BounceBackStreamingMethod<mpcd::PlanarPoreGeometry, mpcd::NoForce>(m);
    mpcd::detail::export_BounceBackStreamingMethod<mpcd::PlanarPoreGeometry, mpcd::SineForce>(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_BounceBackStreamingMethodGPU<mpcd::PlanarPoreGeometry, mpcd::BlockForce>(
        m);
    mpcd::detail::export_BounceBackStreamingMethodGPU<mpcd::PlanarPoreGeometry,
                                                      mpcd::ConstantForce>(m);
    mpcd::detail::export_BounceBackStreamingMethodGPU<mpcd::PlanarPoreGeometry, mpcd::NoForce>(m);
    mpcd::detail::export_BounceBackStreamingMethodGPU<mpcd::PlanarPoreGeometry, mpcd::SineForce>(m);
#endif // ENABLE_HIP

    mpcd::detail::export_BounceBackNVE<mpcd::ParallelPlateGeometry>(m);
    mpcd::detail::export_BounceBackNVE<mpcd::PlanarPoreGeometry>(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_BounceBackNVEGPU<mpcd::ParallelPlateGeometry>(m);
    mpcd::detail::export_BounceBackNVEGPU<mpcd::PlanarPoreGeometry>(m);
#endif // ENABLE_HIP

    mpcd::detail::export_VirtualParticleFiller(m);
    mpcd::detail::export_ManualVirtualParticleFiller(m);
    mpcd::detail::export_ParallelPlateGeometryFiller(m);
    mpcd::detail::export_PlanarPoreGeometryFiller(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_ParallelPlateGeometryFillerGPU(m);
    mpcd::detail::export_PlanarPoreGeometryFillerGPU(m);
#endif // ENABLE_HIP

#ifdef ENABLE_MPI
    mpcd::detail::export_Communicator(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_CommunicatorGPU(m);
#endif // ENABLE_HIP
#endif // ENABLE_MPI
    }
