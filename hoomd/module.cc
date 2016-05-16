// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander All developers are free to add the calls needed to export their modules

#include "HOOMDMath.h"
#include "ExecutionConfiguration.h"
#include "ClockSource.h"
#include "Profiler.h"
#include "ParticleData.h"
#include "SystemDefinition.h"
#include "BondedGroupData.h"
#include "Initializers.h"
#include "HOOMDInitializer.h"
#include "GSDReader.h"
#include "RandomGenerator.h"
#include "Compute.h"
#include "ComputeThermo.h"
#include "CellList.h"
#include "CellListStencil.h"
#include "ForceCompute.h"
#include "ForceConstraint.h"
#include "ConstForceCompute.h"
#include "Analyzer.h"
#include "IMDInterface.h"
#include "HOOMDDumpWriter.h"
#include "POSDumpWriter.h"
#include "PDBDumpWriter.h"
#include "MOL2DumpWriter.h"
#include "DCDDumpWriter.h"
#include "GSDDumpWriter.h"
#include "Logger.h"
#include "MSDAnalyzer.h"
#include "CallbackAnalyzer.h"
#include "Updater.h"
#include "Integrator.h"
#include "SFCPackUpdater.h"
#include "BoxResizeUpdater.h"
#include "System.h"
#include "Variant.h"
#include "Messenger.h"

// include GPU classes
#ifdef ENABLE_CUDA
#include <cuda.h>
#include "CellListGPU.h"
#include "ComputeThermoGPU.h"
#include "SFCPackUpdaterGPU.h"

#include <cuda_profiler_api.h>
#endif

// include MPI classes
#ifdef ENABLE_MPI
#include "Communicator.h"
#include "DomainDecomposition.h"
#include "LoadBalancer.h"

#ifdef ENABLE_CUDA
#include "CommunicatorGPU.h"
#include "LoadBalancerGPU.h"
#endif // ENABLE_CUDA
#endif // ENABLE_MPI

#include "SignalHandler.h"

#include "HOOMDVersion.h"
#include "hoomd/extern/num_util.h"

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;
namespace bnp=boost::python::numeric;

#include <iostream>
#include <sstream>
#include <fstream>
using namespace std;

/*! \file hoomd_module.cc
    \brief Brings all of the export_* functions together to export the hoomd python module
*/

/* numpy is terrible (see /opt/local/Library/Frameworks/Python.framework/Versions/2.7/
lib/python2.7/site-packages/numpy/core/generate_numpy_array.py)
The following #defines help get around this
*/

#if PY_VERSION_HEX >= 0x03000000
#define MY_PY_VER_3x
#else
#define MY_PY_VER_2x
#endif

#ifdef MY_PY_VER_3x
void *my_import_array()
    {
    import_array();
    return NULL;
    }
#endif
#ifdef MY_PY_VER_2x
void my_import_array()
    {
    import_array();
    }
#endif

//! Method for getting the current version of HOOMD
/*! \returns Current HOOMD version identification string
*/
string get_hoomd_version()
    {
    ostringstream ver;
    // always outputting main version number: #402
    ver << "HOOMD-blue " << HOOMD_VERSION << endl;

    return ver.str();
    }

//! Layer for omp_get_num_procs()
int get_num_procs()
    {
    return 1;
    }

//! Get the hoomd version as a tuple
object get_hoomd_version_tuple()
    {
    return boost::python::make_tuple(HOOMD_VERSION_MAJOR, HOOMD_VERSION_MINOR, HOOMD_VERSION_PATCH);
    }

//! Get the CUDA version as a tuple
object get_cuda_version_tuple()
    {
    #ifdef ENABLE_CUDA
    int major = CUDA_VERSION / 1000;
    int minor = CUDA_VERSION / 10 % 100;
    return boost::python::make_tuple(major, minor);
    #else
    return boost::python::make_tuple(0,0);
    #endif
    }

//! Get the compiler version
string get_compiler_version()
    {
    #if defined(__GNUC__) && !(defined(__clang__) || defined(__INTEL_COMPILER))
    ostringstream o;
    o << "gcc " << __GNUC__ << "." << __GNUC_MINOR__ << "." <<  __GNUC_PATCHLEVEL__;
    return o.str();

    #elif defined(__clang__)
    ostringstream o;
    o << "clang " << __clang_major__ << "." << __clang_minor__ << "." <<  __clang_patchlevel__;
    return o.str();

    #elif defined(__INTEL_COMPILER)
    ostringstream o;
    o << "icc " << __INTEL_COMPILER;
    return o.str();

    #else
    return string("unknown");

    #endif
    }

//! Determine availability of MPI support
bool is_MPI_available()
   {
   return
#ifdef ENABLE_MPI
       true;
#else
       false;
#endif
    }

void mpi_barrier_world()
    {
    #ifdef ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
    }

//! Start the CUDA profiler
void cuda_profile_start()
    {
    #ifdef ENABLE_CUDA
    cudaDeviceSynchronize();
    cudaProfilerStart();
    #endif
    }

//! Stop the CUDA profiler
void cuda_profile_stop()
    {
    #ifdef ENABLE_CUDA
    cudaDeviceSynchronize();
    cudaProfilerStop();
    #endif
    }

// values used in measuring hoomd launch timing
unsigned int hoomd_launch_time, hoomd_start_time, hoomd_mpi_init_time;
bool hoomd_launch_timing=false;

#ifdef ENABLE_MPI
//! Environment variables needed for setting up MPI
char env_enable_mpi_cuda[] = "MV2_USE_CUDA=1";

//! Initialize the MPI environment
void initialize_mpi()
    {
    #ifdef ENABLE_MPI_CUDA
    // if we are using an MPI-CUDA implementation, enable this feature
    // before the MPI_Init
    putenv(env_enable_mpi_cuda);
    #endif

    // benchmark hoomd launch times
    if (getenv("HOOMD_LAUNCH_TIME"))
        {
        // get the time that mpirun was called
        hoomd_launch_time = atoi(getenv("HOOMD_LAUNCH_TIME"));

        // compute the number of seconds to get here
        timeval t;
        gettimeofday(&t, NULL);
        hoomd_start_time = t.tv_sec - hoomd_launch_time;
        hoomd_launch_timing = true;
        }

    // initalize MPI
    MPI_Init(0, (char ***) NULL);

    if (hoomd_launch_timing)
        {
        // compute the number of seconds to get past mpi_init
        timeval t;
        gettimeofday(&t, NULL);
        hoomd_mpi_init_time = t.tv_sec - hoomd_launch_time;
        }
    }

//! Get the processor name associated to this rank
string get_mpi_proc_name()
    {
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(proc_name, &name_len);
    return string(proc_name);
    }

//! Finalize MPI environment
void finalize_mpi()
    {
    MPI_Finalize();
    }
#endif

//! Abort MPI runs
void abort_mpi(boost::shared_ptr<ExecutionConfiguration> exec_conf)
    {
    #ifdef ENABLE_MPI
    if(exec_conf->getNRanksGlobal() > 1)
        {
        MPI_Abort(exec_conf->getMPICommunicator(), MPI_ERR_OTHER);
        }
    #endif
    }

//! Create the python module
/*! each class setup their own python exports in a function export_ClassName
    create the hoomd python module and define the exports here.
*/
BOOST_PYTHON_MODULE(_hoomd)
    {
    #ifdef ENABLE_MPI
    // initialize MPI early
    initialize_mpi();

    // register clean-up function
    Py_AtExit(finalize_mpi);
    def("get_mpi_proc_name", get_mpi_proc_name);
    #endif

    // setup needed for numpy
    my_import_array();
    bnp::array::set_module_and_type("numpy", "ndarray");

    def("abort_mpi", abort_mpi);
    def("mpi_barrier_world", mpi_barrier_world);

    def("hoomd_compile_flags", &hoomd_compile_flags);
    def("output_version_info", &output_version_info);
    def("get_hoomd_version", &get_hoomd_version);

    def("get_num_procs", &get_num_procs);
    scope().attr("__version__") = get_hoomd_version_tuple();
    scope().attr("__git_sha1__") = HOOMD_GIT_SHA1;
    scope().attr("__git_refspec__") = HOOMD_GIT_REFSPEC;
    scope().attr("__cuda_version__") = get_cuda_version_tuple();
    scope().attr("__compiler_version__") = get_compiler_version();

    def("is_MPI_available", &is_MPI_available);

    def("cuda_profile_start", &cuda_profile_start);
    def("cuda_profile_stop", &cuda_profile_stop);

    class_<std::vector<Scalar> >("std_vector_scalar")
    .def(vector_indexing_suite<std::vector<Scalar> >())
    ;

    class_< std::vector<unsigned int> >("std_vector_uint")
    .def(vector_indexing_suite<std::vector<unsigned int> >())
    ;

    // data structures
    class_<std::vector<int> >("std_vector_int")
    .def(vector_indexing_suite<std::vector<int> >());

    class_<std::vector<Scalar3> >("std_vector_scalar3")
    .def(vector_indexing_suite<std::vector<Scalar3> >());

    class_<std::vector<Scalar4> >("std_vector_scalar4")
    .def(vector_indexing_suite<std::vector<Scalar4> >());

    InstallSIGINTHandler();

    // utils
    export_hoomd_math_functions();
    export_ClockSource();
    export_Profiler();

    // data structures
    export_BoxDim();
    export_ParticleData();
    export_SnapshotParticleData();
    export_ExecutionConfiguration();
    export_SystemDefinition();
    export_SnapshotSystemData();
    export_BondedGroupData<BondData,Bond>("BondData","BondDataSnapshot");
    export_BondedGroupData<AngleData,Angle>("AngleData","AngleDataSnapshot");
    export_BondedGroupData<DihedralData,Dihedral>("DihedralData","DihedralDataSnapshot");
    export_BondedGroupData<ImproperData,Dihedral>("ImproperData","ImproperDataSnapshot",false);
    export_BondedGroupData<ConstraintData,Constraint>("ConstraintData","ConstraintDataSnapshot");

    // initializers
    export_RandomInitializer();
    export_SimpleCubicInitializer();
    export_HOOMDInitializer();
    export_GSDReader();
    export_RandomGenerator();

    // computes
    export_Compute();
    export_ComputeThermo();
    export_CellList();
    export_CellListStencil();
    export_ForceCompute();
    export_ForceConstraint();
    export_ConstForceCompute();

#ifdef ENABLE_CUDA
    export_CellListGPU();
    export_ComputeThermoGPU();
#endif

    // analyzers
    export_Analyzer();
    export_IMDInterface();
    export_HOOMDDumpWriter();
    export_POSDumpWriter();
    export_PDBDumpWriter();
    export_DCDDumpWriter();
    export_GSDDumpWriter();
    export_MOL2DumpWriter();
    export_Logger();
    export_MSDAnalyzer();
    export_CallbackAnalyzer();
    export_ParticleGroup();

    // updaters
    export_Updater();
    export_Integrator();
    export_BoxResizeUpdater();
    export_SFCPackUpdater();
#ifdef ENABLE_CUDA
    export_SFCPackUpdaterGPU();
#endif

#ifdef ENABLE_MPI
    export_Communicator();
    export_DomainDecomposition();
    export_LoadBalancer();
#ifdef ENABLE_CUDA
    export_CommunicatorGPU();
    export_LoadBalancerGPU();
#endif // ENABLE_CUDA
#endif // ENABLE_MPI

    // system
    export_System();

    // variant
    export_Variant();

    // messenger
    export_Messenger();

    // boost 1.60.0 compatibility
    #if (BOOST_VERSION == 106000)
    register_ptr_to_python< boost::shared_ptr< IMDInterface > >();
    register_ptr_to_python< boost::shared_ptr< DCDDumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< POSDumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< HOOMDDumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< PDBDumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< MOL2DumpWriter > >();
    register_ptr_to_python< boost::shared_ptr< MSDAnalyzer > >();
    register_ptr_to_python< boost::shared_ptr< Logger > >();
    register_ptr_to_python< boost::shared_ptr< CallbackAnalyzer > >();
    register_ptr_to_python< boost::shared_ptr< DomainDecomposition > >();
    register_ptr_to_python< boost::shared_ptr< CellList > >();
    register_ptr_to_python< boost::shared_ptr< CellListStencil > >();
    register_ptr_to_python< boost::shared_ptr< ForceConstraint > >();
    register_ptr_to_python< boost::shared_ptr< ConstForceCompute > >();
    register_ptr_to_python< boost::shared_ptr< ExecutionConfiguration > >();
    register_ptr_to_python< boost::shared_ptr< SystemDefinition > >();
    register_ptr_to_python< boost::shared_ptr< ParticleData > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotParticleData<float> > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotParticleData<double> > >();
    register_ptr_to_python< boost::shared_ptr< RandomGenerator > >();
    register_ptr_to_python< boost::shared_ptr< PolymerParticleGenerator > >();
    register_ptr_to_python< boost::shared_ptr< HOOMDInitializer > >();
    register_ptr_to_python< boost::shared_ptr< ParticleGroup > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelector > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorAll > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorTag > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorType > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorRigid > >();
    register_ptr_to_python< boost::shared_ptr< ParticleSelectorCuboid > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotSystemData<float> > >();
    register_ptr_to_python< boost::shared_ptr< SnapshotSystemData<double> > >();
    register_ptr_to_python< boost::shared_ptr< System > >();
    register_ptr_to_python< boost::shared_ptr< Integrator > >();
    register_ptr_to_python< boost::shared_ptr< SFCPackUpdater > >();
    register_ptr_to_python< boost::shared_ptr< double2 > >();
    register_ptr_to_python< boost::shared_ptr< double3 > >();
    register_ptr_to_python< boost::shared_ptr< double4 > >();
    register_ptr_to_python< boost::shared_ptr< float2 > >();
    register_ptr_to_python< boost::shared_ptr< float3 > >();
    register_ptr_to_python< boost::shared_ptr< float4 > >();
    register_ptr_to_python< boost::shared_ptr< uint2 > >();
    register_ptr_to_python< boost::shared_ptr< uint3 > >();
    register_ptr_to_python< boost::shared_ptr< uint4 > >();
    register_ptr_to_python< boost::shared_ptr< int2 > >();
    register_ptr_to_python< boost::shared_ptr< int3 > >();
    register_ptr_to_python< boost::shared_ptr< int4 > >();
    register_ptr_to_python< boost::shared_ptr< char3 > >();
    register_ptr_to_python< boost::shared_ptr< Variant > >();
    register_ptr_to_python< boost::shared_ptr< VariantConst > >();
    register_ptr_to_python< boost::shared_ptr< VariantLinear > >();
    register_ptr_to_python< boost::shared_ptr< Messenger > >();

    #ifdef ENABLE_CUDA
    #ifdef ENABLE_MPI
    register_ptr_to_python< boost::shared_ptr< LoadBalancerGPU > >();
    register_ptr_to_python< boost::shared_ptr< CommunicatorGPU > >();
    #endif
    register_ptr_to_python< boost::shared_ptr< CellListGPU > >();
    #endif

    #ifdef ENABLE_MPI
    register_ptr_to_python< boost::shared_ptr< Communicator > >();
    register_ptr_to_python< boost::shared_ptr< LoadBalancer > >();
    #endif
    #endif
    }
