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
                   ComputeThermo.cc
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
                   LoadBalancer.cc
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
                   extern/BVLSSolver.cc
                   extern/gsd.c
                   extern/imd.cc
                   extern/kiss_fft.cc
                   extern/kiss_fftnd.cc
                   extern/mgpuutil.cc
                   extern/num_util.cc
                   extern/vmdsock.cc
                   extern/xmlParser.cc
                   )

if (ENABLE_CUDA)
list(APPEND _hoomd_sources CellListGPU.cc
                           CommunicatorGPU.cc
                           ComputeThermoGPU.cc
                           LoadBalancerGPU.cc
                           SFCPackUpdaterGPU.cc
                           )
endif()

set(_hoomd_cu_sources BondedGroupData.cu
                      CellListGPU.cu
                      CommunicatorGPU.cu
                      ComputeThermoGPU.cu
                      Integrator.cu
                      LoadBalancerGPU.cu
                      ParticleData.cu
                      ParticleGroup.cu
                      SFCPackUpdaterGPU.cu
                      extern/mgpucontext.cu)

# Need to define NO_IMPORT_ARRAY in every file but module.cc
set_source_files_properties(${_hoomd_sources} ${_hoomd_cu_sources} PROPERTIES COMPILE_DEFINITIONS NO_IMPORT_ARRAY)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_hoomd_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

add_library (_hoomd SHARED module.cc ${_hoomd_sources} ${_CUDA_GENERATED_FILES})

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
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/${file}"
    )
ENDMACRO(copy_file)

set(files analyze.py
          benchmark.py
          cite.py
          comm.py
          compute.py
          context.py
          data.py
          dump.py
          group.py
          __init__.py
          init.py
          integrate.py
          meta.py
          option.py
          sorter.py
          update.py
          util.py
          variant.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_hoomd ALL DEPENDS ${files})

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()

option(BUILD_MD "Build the md package" on)
if (BUILD_MD)
    if (ENABLE_MPI)
        # add the distributed FFT library
        add_subdirectory(${HOOMD_SOURCE_DIR}/hoomd/extern/dfftlib)
    endif()

    add_subdirectory(md)
endif()

if (NOT SINGLE_PRECISION)
option(BUILD_HPMC "Build the hpmc package" on)
else ()
option(BUILD_HPMC "Build the hpmc package" off)
endif()

if (NOT SINGLE_PRECISION AND BUILD_HPMC)
    add_subdirectory(hpmc)
endif()
