// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
void export_ATCollisionMethod(pybind11::module&);
void export_BlockForce(pybind11::module&);
void export_CellList(pybind11::module&);
void export_CellThermoCompute(pybind11::module&);
void export_CollisionMethod(pybind11::module&);
#ifdef ENABLE_MPI
void export_Communicator(pybind11::module&);
#endif // ENABLE_MPI
void export_ConstantForce(pybind11::module&);
void export_CosineChannelGeometry(pybind11::module&);
void export_CosineChannelGeometryFiller(pybind11::module&);
void export_CosineExpansionContractionGeometry(pybind11::module&);
void export_CosineExpansionContractionGeometryFiller(pybind11::module&);
void export_Integrator(pybind11::module&);
void export_ManualVirtualParticleFiller(pybind11::module&);
void export_NoForce(pybind11::module&);
void export_ParallelPlateGeometry(pybind11::module&);
void export_ParallelPlateGeometryFiller(pybind11::module&);
void export_PlanarPoreGeometry(pybind11::module&);
void export_PlanarPoreGeometryFiller(pybind11::module&);
void export_Sorter(pybind11::module&);
void export_SphereGeometry(pybind11::module&);
void export_SphereGeometryFiller(pybind11::module&);
void export_SineForce(pybind11::module&);
void export_SRDCollisionMethod(pybind11::module&);
void export_StreamingMethod(pybind11::module&);
void export_VirtualParticleFiller(pybind11::module&);
#ifdef ENABLE_HIP
void export_ATCollisionMethodGPU(pybind11::module&);
void export_CellListGPU(pybind11::module&);
void export_CellThermoComputeGPU(pybind11::module&);
#ifdef ENABLE_MPI
void export_CommunicatorGPU(pybind11::module&);
#endif // ENABLE_MPI
void export_CosineChannelGeometryFillerGPU(pybind11::module&);
void export_CosineExpansionContractionGeometryFillerGPU(pybind11::module&);
void export_ParallelPlateGeometryFillerGPU(pybind11::module&);
void export_PlanarPoreGeometryFillerGPU(pybind11::module&);
void export_SorterGPU(pybind11::module&);
void export_SphereGeometryFillerGPU(pybind11::module&);
void export_SRDCollisionMethodGPU(pybind11::module&);
#endif // ENABLE_HIP

void export_BulkStreamingMethodBlockForce(pybind11::module&);
void export_BulkStreamingMethodConstantForce(pybind11::module&);
void export_BulkStreamingMethodNoForce(pybind11::module&);
void export_BulkStreamingMethodSineForce(pybind11::module&);
#ifdef ENABLE_HIP
void export_BulkStreamingMethodBlockForceGPU(pybind11::module&);
void export_BulkStreamingMethodConstantForceGPU(pybind11::module&);
void export_BulkStreamingMethodNoForceGPU(pybind11::module&);
void export_BulkStreamingMethodSineForceGPU(pybind11::module&);
#endif // ENABLE_HIP

// cosine channel
void export_BounceBackStreamingMethodCosineChannelGeometryBlockForce(pybind11::module&);
void export_BounceBackStreamingMethodCosineChannelGeometryConstantForce(pybind11::module&);
void export_BounceBackStreamingMethodCosineChannelGeometryNoForce(pybind11::module&);
void export_BounceBackStreamingMethodCosineChannelGeometrySineForce(pybind11::module&);
// cosine expansion contraction
void export_BounceBackStreamingMethodCosineExpansionContractionGeometryBlockForce(
    pybind11::module&);
void export_BounceBackStreamingMethodCosineExpansionContractionGeometryConstantForce(
    pybind11::module&);
void export_BounceBackStreamingMethodCosineExpansionContractionGeometryNoForce(pybind11::module&);
void export_BounceBackStreamingMethodCosineExpansionContractionGeometrySineForce(pybind11::module&);
// parallel plate
void export_BounceBackStreamingMethodParallelPlateGeometryBlockForce(pybind11::module&);
void export_BounceBackStreamingMethodParallelPlateGeometryConstantForce(pybind11::module&);
void export_BounceBackStreamingMethodParallelPlateGeometryNoForce(pybind11::module&);
void export_BounceBackStreamingMethodParallelPlateGeometrySineForce(pybind11::module&);
// planar pore
void export_BounceBackStreamingMethodPlanarPoreGeometryBlockForce(pybind11::module&);
void export_BounceBackStreamingMethodPlanarPoreGeometryConstantForce(pybind11::module&);
void export_BounceBackStreamingMethodPlanarPoreGeometryNoForce(pybind11::module&);
void export_BounceBackStreamingMethodPlanarPoreGeometrySineForce(pybind11::module&);
// sphere
void export_BounceBackStreamingMethodSphereGeometryBlockForce(pybind11::module&);
void export_BounceBackStreamingMethodSphereGeometryConstantForce(pybind11::module&);
void export_BounceBackStreamingMethodSphereGeometryNoForce(pybind11::module&);
void export_BounceBackStreamingMethodSphereGeometrySineForce(pybind11::module&);
#ifdef ENABLE_HIP
// cosine channel
void export_BounceBackStreamingMethodCosineChannelGeometryBlockForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodCosineChannelGeometryConstantForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodCosineChannelGeometryNoForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodCosineChannelGeometrySineForceGPU(pybind11::module&);
// cosine expansion contraction
void export_BounceBackStreamingMethodCosineExpansionContractionGeometryBlockForceGPU(
    pybind11::module&);
void export_BounceBackStreamingMethodCosineExpansionContractionGeometryConstantForceGPU(
    pybind11::module&);
void export_BounceBackStreamingMethodCosineExpansionContractionGeometryNoForceGPU(
    pybind11::module&);
void export_BounceBackStreamingMethodCosineExpansionContractionGeometrySineForceGPU(
    pybind11::module&);
// parallel plate
void export_BounceBackStreamingMethodParallelPlateGeometryBlockForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodParallelPlateGeometryConstantForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodParallelPlateGeometryNoForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodParallelPlateGeometrySineForceGPU(pybind11::module&);
// planar pore
void export_BounceBackStreamingMethodPlanarPoreGeometryBlockForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodPlanarPoreGeometryConstantForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodPlanarPoreGeometryNoForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodPlanarPoreGeometrySineForceGPU(pybind11::module&);
// sphere
void export_BounceBackStreamingMethodSphereGeometryBlockForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodSphereGeometryConstantForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodSphereGeometryNoForceGPU(pybind11::module&);
void export_BounceBackStreamingMethodSphereGeometrySineForceGPU(pybind11::module&);
#endif // ENABLE_HIP

void export_BounceBackNVECosineChannelGeometry(pybind11::module&);
void export_BounceBackNVECosineExpansionContractionGeometry(pybind11::module&);
void export_BounceBackNVEParallelPlateGeometry(pybind11::module&);
void export_BounceBackNVEPlanarPoreGeometry(pybind11::module&);
void export_BounceBackNVESphereGeometry(pybind11::module&);
#ifdef ENABLE_HIP
void export_BounceBackNVECosineChannelGeometryGPU(pybind11::module&);
void export_BounceBackNVECosineExpansionContractionGeometryGPU(pybind11::module&);
void export_BounceBackNVEParallelPlateGeometryGPU(pybind11::module&);
void export_BounceBackNVEPlanarPoreGeometryGPU(pybind11::module&);
void export_BounceBackNVESphereGeometryGPU(pybind11::module&);
#endif // ENABLE_HIP

    } // end namespace detail

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
using namespace hoomd::mpcd;
using namespace hoomd::mpcd::detail;

PYBIND11_MODULE(_mpcd, m)
    {
    // base classes must come first
    export_CollisionMethod(m);
    export_VirtualParticleFiller(m);
    export_StreamingMethod(m);

    export_ATCollisionMethod(m);
    export_BlockForce(m);
    export_CellList(m);
    export_CellThermoCompute(m);
#ifdef ENABLE_MPI
    export_Communicator(m);
#endif // ENABLE_MPI
    export_ConstantForce(m);
    export_CosineChannelGeometry(m);
    export_CosineChannelGeometryFiller(m);
    export_CosineExpansionContractionGeometry(m);
    export_CosineExpansionContractionGeometryFiller(m);
    export_Integrator(m);
    export_ManualVirtualParticleFiller(m);
    export_NoForce(m);
    export_ParallelPlateGeometry(m);
    export_ParallelPlateGeometryFiller(m);
    export_PlanarPoreGeometry(m);
    export_PlanarPoreGeometryFiller(m);
    export_Sorter(m);
    export_SphereGeometry(m);
    export_SphereGeometryFiller(m);
    export_SineForce(m);
    export_SRDCollisionMethod(m);
#ifdef ENABLE_HIP
    export_ATCollisionMethodGPU(m);
    export_CellListGPU(m);
    export_CellThermoComputeGPU(m);
#ifdef ENABLE_MPI
    export_CommunicatorGPU(m);
#endif // ENABLE_MPI
    export_CosineChannelGeometryFillerGPU(m);
    export_CosineExpansionContractionGeometryFillerGPU(m);
    export_ParallelPlateGeometryFillerGPU(m);
    export_PlanarPoreGeometryFillerGPU(m);
    export_SorterGPU(m);
    export_SphereGeometryFillerGPU(m);
    export_SRDCollisionMethodGPU(m);
#endif // ENABLE_HIP

    export_BulkStreamingMethodBlockForce(m);
    export_BulkStreamingMethodConstantForce(m);
    export_BulkStreamingMethodNoForce(m);
    export_BulkStreamingMethodSineForce(m);
#ifdef ENABLE_HIP
    export_BulkStreamingMethodBlockForceGPU(m);
    export_BulkStreamingMethodConstantForceGPU(m);
    export_BulkStreamingMethodNoForceGPU(m);
    export_BulkStreamingMethodSineForceGPU(m);
#endif // ENABLE_HIP

    // cosine channel
    export_BounceBackStreamingMethodCosineChannelGeometryBlockForce(m);
    export_BounceBackStreamingMethodCosineChannelGeometryConstantForce(m);
    export_BounceBackStreamingMethodCosineChannelGeometryNoForce(m);
    export_BounceBackStreamingMethodCosineChannelGeometrySineForce(m);
    // cosine expansion contraction
    export_BounceBackStreamingMethodCosineExpansionContractionGeometryBlockForce(m);
    export_BounceBackStreamingMethodCosineExpansionContractionGeometryConstantForce(m);
    export_BounceBackStreamingMethodCosineExpansionContractionGeometryNoForce(m);
    export_BounceBackStreamingMethodCosineExpansionContractionGeometrySineForce(m);
    // parallel plate
    export_BounceBackStreamingMethodParallelPlateGeometryBlockForce(m);
    export_BounceBackStreamingMethodParallelPlateGeometryConstantForce(m);
    export_BounceBackStreamingMethodParallelPlateGeometryNoForce(m);
    export_BounceBackStreamingMethodParallelPlateGeometrySineForce(m);
    // planar pore
    export_BounceBackStreamingMethodPlanarPoreGeometryBlockForce(m);
    export_BounceBackStreamingMethodPlanarPoreGeometryConstantForce(m);
    export_BounceBackStreamingMethodPlanarPoreGeometryNoForce(m);
    export_BounceBackStreamingMethodPlanarPoreGeometrySineForce(m);
    // sphere
    export_BounceBackStreamingMethodSphereGeometryBlockForce(m);
    export_BounceBackStreamingMethodSphereGeometryConstantForce(m);
    export_BounceBackStreamingMethodSphereGeometryNoForce(m);
    export_BounceBackStreamingMethodSphereGeometrySineForce(m);
#ifdef ENABLE_HIP
    // cosine channel
    export_BounceBackStreamingMethodCosineChannelGeometryBlockForceGPU(m);
    export_BounceBackStreamingMethodCosineChannelGeometryConstantForceGPU(m);
    export_BounceBackStreamingMethodCosineChannelGeometryNoForceGPU(m);
    export_BounceBackStreamingMethodCosineChannelGeometrySineForceGPU(m);
    // cosine expansion contraction
    export_BounceBackStreamingMethodCosineExpansionContractionGeometryBlockForceGPU(m);
    export_BounceBackStreamingMethodCosineExpansionContractionGeometryConstantForceGPU(m);
    export_BounceBackStreamingMethodCosineExpansionContractionGeometryNoForceGPU(m);
    export_BounceBackStreamingMethodCosineExpansionContractionGeometrySineForceGPU(m);
    // parallel plate
    export_BounceBackStreamingMethodParallelPlateGeometryBlockForceGPU(m);
    export_BounceBackStreamingMethodParallelPlateGeometryConstantForceGPU(m);
    export_BounceBackStreamingMethodParallelPlateGeometryNoForceGPU(m);
    export_BounceBackStreamingMethodParallelPlateGeometrySineForceGPU(m);
    // planar pore
    export_BounceBackStreamingMethodPlanarPoreGeometryBlockForceGPU(m);
    export_BounceBackStreamingMethodPlanarPoreGeometryConstantForceGPU(m);
    export_BounceBackStreamingMethodPlanarPoreGeometryNoForceGPU(m);
    export_BounceBackStreamingMethodPlanarPoreGeometrySineForceGPU(m);
    // planar pore
    export_BounceBackStreamingMethodSphereGeometryBlockForceGPU(m);
    export_BounceBackStreamingMethodSphereGeometryConstantForceGPU(m);
    export_BounceBackStreamingMethodSphereGeometryNoForceGPU(m);
    export_BounceBackStreamingMethodSphereGeometrySineForceGPU(m);
#endif // ENABLE_HIP

    export_BounceBackNVECosineChannelGeometry(m);
    export_BounceBackNVECosineExpansionContractionGeometry(m);
    export_BounceBackNVEParallelPlateGeometry(m);
    export_BounceBackNVEPlanarPoreGeometry(m);
    export_BounceBackNVESphereGeometry(m);
#ifdef ENABLE_HIP
    export_BounceBackNVECosineChannelGeometryGPU(m);
    export_BounceBackNVECosineExpansionContractionGeometryGPU(m);
    export_BounceBackNVEParallelPlateGeometryGPU(m);
    export_BounceBackNVEPlanarPoreGeometryGPU(m);
    export_BounceBackNVESphereGeometryGPU(m);
#endif // ENABLE_HIP
    }
