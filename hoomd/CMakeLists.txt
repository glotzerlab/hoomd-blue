# Maintainer: joaander

# configure the version information file
configure_file (HOOMDVersion.h.in ${HOOMD_BINARY_DIR}/include/HOOMDVersion.h)

set(_hoomd_sources Analyzer.cc
                   Autotuner.cc
                   BondedGroupData.cc
                   BoxResizeUpdater.cc
                   CallbackAnalyzer.cc
                   CellList.cc
                   CellListStencil.cc
                   ClockSource.cc
                   Communicator.cc
                   CommunicatorGPU.cc
                   Compute.cc
                   ConstForceCompute.cc
                   DCDDumpWriter.cc
                   DomainDecomposition.cc
                   ExecutionConfiguration.cc
                   ForceCompute.cc
                   ForceConstraint.cc
                   GSDDumpWriter.cc
                   GSDReader.cc
                   HOOMDDumpWriter.cc
                   HOOMDInitializer.cc
                   HOOMDMath.cc
                   HOOMDVersion.cc
                   IMDInterface.cc
                   Initializers.cc
                   Integrator.cc
                   IntegratorData.cc
                   Logger.cc
                   Messenger.cc
                   MOL2DumpWriter.cc
                   MSDAnalyzer.cc
                   ParticleData.cc
                   ParticleGroup.cc
                   PDBDumpWriter.cc
                   POSDumpWriter.cc
                   Profiler.cc
                   RandomGenerator.cc
                   SFCPackUpdater.cc
                   SignalHandler.cc
                   SnapshotSystemData.cc
                   System.cc
                   SystemDefinition.cc
                   Updater.cc
                   Variant.cc
                   )

set(_hoomd_cu_sources "")

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_hoomd_sources} ${_hoomd_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_hoomd_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_hoomd ${_hoomd_sources} ${_CUDA_GENERATED_FILES} module.cc)

# link the library to its dependencies
target_link_libraries(_hoomd ${HOOMD_COMMON_LIBS})

# if we are compiling with MPI support built in, set appropriate
# compiler/linker flags
if (ENABLE_MPI)
   if(MPI_COMPILE_FLAGS)
       set_target_properties(_hoomd PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
   endif(MPI_COMPILE_FLAGS)
   if(MPI_LINK_FLAGS)
       set_target_properties(_hoomd PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
   endif(MPI_LINK_FLAGS)
endif(ENABLE_MPI)

##################################
# change the name of the library to be a valid python module
# tweak the properties of the output to make a functional python module
set_target_properties(_hoomd PROPERTIES PREFIX "" OUTPUT_NAME "_hoomd")

# .dylib is not recognized as a python module by python on Mac OS X
if(APPLE)
    set_target_properties(_hoomd PROPERTIES SUFFIX ".so")
endif(APPLE)
fix_cudart_rpath(_hoomd)

# install the library
install(TARGETS _hoomd
        LIBRARY DESTINATION ${_hoomd_so_install_dir}
        ARCHIVE DESTINATION ${LIB_BASE_INSTALL_DIR}
        )
