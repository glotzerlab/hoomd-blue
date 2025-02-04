// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Action.h"
#include "Analyzer.h"
#include "BondedGroupData.h"
#include "BoxResizeUpdater.h"
#include "CellList.h"
#include "CellListStencil.h"
#include "ClockSource.h"
#include "Compute.h"
#include "DCDDumpWriter.h"
#include "ExecutionConfiguration.h"
#include "ForceCompute.h"
#include "ForceConstraint.h"
#include "GSDDequeWriter.h"
#include "GSDDumpWriter.h"
#include "GSDReader.h"
#include "HOOMDMath.h"
#include "Initializers.h"
#include "Integrator.h"
#include "LoadBalancer.h"
#include "MeshDefinition.h"
#include "MeshGroupData.h"
#include "Messenger.h"
#include "ParticleData.h"
#include "ParticleFilterUpdater.h"
#include "PythonAnalyzer.h"
#include "PythonLocalDataAccess.h"
#include "PythonTuner.h"
#include "PythonUpdater.h"
#include "SFCPackTuner.h"
#include "SnapshotSystemData.h"
#include "System.h"
#include "SystemDefinition.h"
#include "Trigger.h"
#include "Tuner.h"
#include "Updater.h"
#include "UpdaterRemoveDrift.h"
#include "Variant.h"
#include "VectorVariant.h"

// ParticleFilter objects
#include "filter/export_filters.h"

// optional MPCD classes
#ifdef BUILD_MPCD
namespace hoomd
    {
namespace mpcd
    {
namespace detail
    {
void export_ParticleData(pybind11::module& pybind11);
void export_ParticleDataSnapshot(pybind11::module& pybind11);
    } // namespace detail
    } // namespace mpcd
    } // namespace hoomd
#endif

// include GPU classes
#ifdef ENABLE_HIP
#include "BoxResizeUpdaterGPU.h"
#include "CellListGPU.h"
#include "LoadBalancerGPU.h"
#include "SFCPackTunerGPU.h"
#include <hip/hip_runtime.h>
#endif

// include MPI classes
#ifdef ENABLE_MPI
#include "Communicator.h"
#include "DomainDecomposition.h"

#ifdef ENABLE_HIP
#include "CommunicatorGPU.h"
#endif // ENABLE_HIP
#endif // ENABLE_MPI

#include "HOOMDVersion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

#include <fstream>
#include <iostream>
#include <sstream>

#ifdef ENABLE_TBB
#include <tbb/task_arena.h>
#endif

/*! \file hoomd_module.cc
    \brief Brings all of the export_* functions together to export the hoomd python module
*/

namespace hoomd
    {
namespace detail
    {
void mpi_barrier_world()
    {
#ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    }

#ifdef ENABLE_MPI
//! Environment variables needed for setting up MPI
char env_enable_mpi_cuda[] = "MV2_USE_CUDA=1";

//! Initialize the MPI environment
int initialize_mpi()
    {
#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_HCC__)
    hipInit(0);
#endif

    // initialize MPI if it has not been initialized by another program
    int external_init = 0;
    MPI_Initialized(&external_init);
    if (!external_init)
        {
        MPI_Init(0, (char***)NULL);
        }

    return external_init;
    }

//! Get the processor name associated to this rank
std::string get_mpi_proc_name()
    {
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(proc_name, &name_len);
    return std::string(proc_name);
    }

//! Finalize MPI environment
void finalize_mpi()
    {
    MPI_Finalize();
    }
#endif

//! Abort MPI runs
void abort_mpi(std::shared_ptr<MPIConfiguration> mpi_conf, int errorcode)
    {
#ifdef ENABLE_MPI
    if (mpi_conf->getNRanksGlobal() > 1)
        {
        // delay for a moment to give time for error messages to print
        Sleep(1000);
        MPI_Abort(mpi_conf->getCommunicator(), errorcode);
        }
#endif
    }

//! broadcast string from root rank to all other ranks
std::string mpi_bcast_str(pybind11::object string,
                          std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    std::string s = pybind11::str(string).cast<std::string>();
#ifdef ENABLE_MPI
    std::string result = s;
    bcast(result, 0, exec_conf->getMPICommunicator());
    return result;
#else
    return s;
#endif
    }

bool mpi_allreduce_bcast_and(bool v, std::shared_ptr<ExecutionConfiguration> exec_conf)
    {
#ifdef ENABLE_MPI
    bool reduced_result = false;
    MPI_Allreduce(&v, &reduced_result, 1, MPI_C_BOOL, MPI_LAND, exec_conf->getMPICommunicator());
    return reduced_result;
#else
    return v;
#endif
    }

    } // end namespace detail

    } // end namespace hoomd

using namespace std;
using namespace hoomd;
using namespace hoomd::detail;

//! Create the python module
/*! each class sets up its own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
PYBIND11_MODULE(_hoomd, m)
    {
#ifdef ENABLE_MPI
    // initialize MPI early, unless already initialized by another program
    int external_init = initialize_mpi();

    // if HOOMD called MPI_Init, it should call MPI_Finalize at exit
    if (!external_init)
        {
        Py_AtExit(finalize_mpi);
        }
    m.def("get_mpi_proc_name", get_mpi_proc_name);
#endif

    m.def("abort_mpi", abort_mpi);
    m.def("mpi_barrier_world", mpi_barrier_world);
    m.def("mpi_bcast_str", mpi_bcast_str);
    m.def("mpi_allreduce_bcast_and", mpi_allreduce_bcast_and);

    pybind11::class_<BuildInfo>(m, "BuildInfo")
        .def_static("getVersion", BuildInfo::getVersion)
        .def_static("getCompileFlags", BuildInfo::getCompileFlags)
        .def_static("getEnableGPU", BuildInfo::getEnableGPU)
        .def_static("getGPUAPIVersion", BuildInfo::getGPUAPIVersion)
        .def_static("getGPUPlatform", BuildInfo::getGPUPlatform)
        .def_static("getCXXCompiler", BuildInfo::getCXXCompiler)
        .def_static("getEnableTBB", BuildInfo::getEnableTBB)
        .def_static("getEnableMPI", BuildInfo::getEnableMPI)
        .def_static("getSourceDir", BuildInfo::getSourceDir)
        .def_static("getInstallDir", BuildInfo::getInstallDir)
        .def_static("getFloatingPointPrecision", BuildInfo::getFloatingPointPrecision);

    pybind11::bind_vector<std::vector<Scalar>>(m, "std_vector_scalar");
    pybind11::bind_vector<std::vector<string>>(m, "std_vector_string");
    pybind11::bind_vector<std::vector<unsigned int>>(m, "std_vector_uint");
    pybind11::bind_vector<std::vector<std::pair<unsigned int, unsigned int>>>(
        m,
        "std_vector_uint_pair");
    pybind11::bind_vector<std::vector<int>>(m, "std_vector_int");
    pybind11::bind_vector<std::vector<Scalar3>>(m, "std_vector_scalar3");
    pybind11::bind_vector<std::vector<Scalar4>>(m, "std_vector_scalar4");

    // utils
    export_hoomd_math_functions(m);
    export_ClockSource(m);

    // data structures
    export_HOOMDHostBuffer(m);
    export_GhostDataFlag(m);
#if ENABLE_HIP
    export_HOOMDDeviceBuffer(m);
#endif
    export_BoxDim(m);
    export_ParticleData(m);
    export_SnapshotParticleData(m);
    export_LocalParticleData<HOOMDHostBuffer>(m, "LocalParticleDataHost");
#if ENABLE_HIP
    export_LocalParticleData<HOOMDDeviceBuffer>(m, "LocalParticleDataDevice");
#endif
    export_MPIConfiguration(m);
    export_ExecutionConfiguration(m);
    export_SystemDefinition(m);
    export_MeshDefinition(m);
    export_SnapshotSystemData(m);
    export_BondedGroupData<BondData, Bond>(m, "BondData", "BondDataSnapshot");
    export_BondedGroupData<AngleData, Angle>(m, "AngleData", "AngleDataSnapshot");
    export_BondedGroupData<TriangleData, Angle>(m, "TriangleData", "TriangleDataSnapshot", false);
    export_BondedGroupData<DihedralData, Dihedral>(m, "DihedralData", "DihedralDataSnapshot");
    export_MeshGroupData<MeshBondData, MeshBond>(m, "MeshBondData", "MeshBondDataSnapshot");
    export_BondedGroupData<ImproperData, Dihedral>(m,
                                                   "ImproperData",
                                                   "ImproperDataSnapshot",
                                                   false);
    export_BondedGroupData<ConstraintData, Constraint>(m,
                                                       "ConstraintData",
                                                       "ConstraintDataSnapshot");
    export_BondedGroupData<PairData, Bond>(m, "PairData", "PairDataSnapshot", false);

    export_LocalGroupData<HOOMDHostBuffer, BondData>(m, "LocalBondDataHost");
    export_LocalGroupData<HOOMDHostBuffer, AngleData>(m, "LocalAngleDataHost");
    export_LocalGroupData<HOOMDHostBuffer, TriangleData>(m, "LocalTriangleDataHost");
    export_LocalGroupData<HOOMDHostBuffer, DihedralData>(m, "LocalDihedralDataHost");
    export_LocalGroupData<HOOMDHostBuffer, MeshBondData>(m, "LocalMeshBondDataHost");
    export_LocalGroupData<HOOMDHostBuffer, ImproperData>(m, "LocalImproperDataHost");
    export_LocalGroupData<HOOMDHostBuffer, ConstraintData>(m, "LocalConstraintDataHost");
    export_LocalGroupData<HOOMDHostBuffer, PairData>(m, "LocalPairDataHost");
#if ENABLE_HIP
    export_LocalGroupData<HOOMDDeviceBuffer, BondData>(m, "LocalBondDataDevice");
    export_LocalGroupData<HOOMDDeviceBuffer, TriangleData>(m, "LocalTriangleDataDevice");
    export_LocalGroupData<HOOMDDeviceBuffer, AngleData>(m, "LocalAngleDataDevice");
    export_LocalGroupData<HOOMDDeviceBuffer, DihedralData>(m, "LocalDihedralDataDevice");
    export_LocalGroupData<HOOMDDeviceBuffer, MeshBondData>(m, "LocalMeshBondDataDevice");
    export_LocalGroupData<HOOMDDeviceBuffer, ImproperData>(m, "LocalImproperDataDevice");
    export_LocalGroupData<HOOMDDeviceBuffer, ConstraintData>(m, "LocalConstraintDataDevice");
    export_LocalGroupData<HOOMDDeviceBuffer, PairData>(m, "LocalPairDataDevice");
#endif
#ifdef BUILD_MPCD
    mpcd::detail::export_ParticleData(m);
    mpcd::detail::export_ParticleDataSnapshot(m);
#endif

    // initializers
    export_GSDReader(m);

    // computes
    export_Autotuned(m);
    export_Action(m);
    export_Compute(m);
    export_CellList(m);
    export_CellListStencil(m);
    export_ForceCompute(m);
    export_LocalForceComputeData<HOOMDHostBuffer>(m, "LocalForceComputeDataHost");
#ifdef ENABLE_HIP
    export_LocalForceComputeData<HOOMDDeviceBuffer>(m, "LocalForceComputeDataDevice");
#endif
    export_ForceConstraint(m);

#ifdef ENABLE_HIP
    export_CellListGPU(m);
#endif

    // analyzers
    export_Analyzer(m);
    export_PythonAnalyzer(m);
    export_DCDDumpWriter(m);
    export_GSDDumpWriter(m);
    export_GSDDequeWriter(m);

    // updaters
    export_Updater(m);
    export_PythonUpdater(m);
    export_Integrator(m);
    export_BoxResizeUpdater(m);
    export_UpdaterRemoveDrift(m);
#ifdef ENABLE_HIP
    export_BoxResizeUpdaterGPU(m);
#endif

    // tuners
    export_Tuner(m);
    export_PythonTuner(m);
    export_SFCPackTuner(m);
    export_LoadBalancer(m);
#ifdef ENABLE_HIP
    export_SFCPackTunerGPU(m);
    export_LoadBalancerGPU(m);
#endif

#ifdef ENABLE_MPI
    export_Communicator(m);
    export_DomainDecomposition(m);
#ifdef ENABLE_HIP
    export_CommunicatorGPU(m);
#endif // ENABLE_HIP
#endif // ENABLE_MPI

    // system
    export_System(m);

    // filters and groups
    export_ParticleFilters(m);
    export_ParticleGroup(m);
    export_ParticleFilterUpdater(m);

    // trigger
    export_Trigger(m);

    // variant
    export_Variant(m);

    // vector variant
    export_VectorVariantBoxClasses(m);

    // messenger
    export_Messenger(m);
    }
