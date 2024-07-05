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
#include "ConfinedStreamingMethod.h"
#include "StreamingGeometry.h"
#include "StreamingMethod.h"
#ifdef ENABLE_HIP
#include "ConfinedStreamingMethodGPU.h"
#endif // ENABLE_HIP

// integration methods
#include "BounceBackNVE.h"
#ifdef ENABLE_HIP
#include "BounceBackNVEGPU.h"
#endif

// virtual particle fillers
#include "SlitGeometryFiller.h"
#include "SlitPoreGeometryFiller.h"
#include "VirtualParticleFiller.h"
#ifdef ENABLE_HIP
#include "SlitGeometryFillerGPU.h"
#include "SlitPoreGeometryFillerGPU.h"
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

    mpcd::detail::export_Integrator(m);

    mpcd::detail::export_CollisionMethod(m);
    mpcd::detail::export_ATCollisionMethod(m);
    mpcd::detail::export_SRDCollisionMethod(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_ATCollisionMethodGPU(m);
    mpcd::detail::export_SRDCollisionMethodGPU(m);
#endif // ENABLE_HIP

    mpcd::detail::export_boundary(m);
    mpcd::detail::export_BulkGeometry(m);
    mpcd::detail::export_SlitGeometry(m);
    mpcd::detail::export_SlitPoreGeometry(m);

    mpcd::detail::export_StreamingMethod(m);
    mpcd::detail::export_ExternalFieldPolymorph(m);
    mpcd::detail::export_ConfinedStreamingMethod<mpcd::detail::BulkGeometry>(m);
    mpcd::detail::export_ConfinedStreamingMethod<mpcd::detail::SlitGeometry>(m);
    mpcd::detail::export_ConfinedStreamingMethod<mpcd::detail::SlitPoreGeometry>(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_ConfinedStreamingMethodGPU<mpcd::detail::BulkGeometry>(m);
    mpcd::detail::export_ConfinedStreamingMethodGPU<mpcd::detail::SlitGeometry>(m);
    mpcd::detail::export_ConfinedStreamingMethodGPU<mpcd::detail::SlitPoreGeometry>(m);
#endif // ENABLE_HIP

    mpcd::detail::export_BounceBackNVE<mpcd::detail::SlitGeometry>(m);
    mpcd::detail::export_BounceBackNVE<mpcd::detail::SlitPoreGeometry>(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_BounceBackNVEGPU<mpcd::detail::SlitGeometry>(m);
    mpcd::detail::export_BounceBackNVEGPU<mpcd::detail::SlitPoreGeometry>(m);
#endif // ENABLE_HIP

    mpcd::detail::export_VirtualParticleFiller(m);
    mpcd::detail::export_SlitGeometryFiller(m);
    mpcd::detail::export_SlitPoreGeometryFiller(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_SlitGeometryFillerGPU(m);
    mpcd::detail::export_SlitPoreGeometryFillerGPU(m);
#endif // ENABLE_HIP

#ifdef ENABLE_MPI
    mpcd::detail::export_Communicator(m);
#ifdef ENABLE_HIP
    mpcd::detail::export_CommunicatorGPU(m);
#endif // ENABLE_HIP
#endif // ENABLE_MPI
    }
