# Maintainer: joaander

add_subdirectory(extern)

# configure the version information file
configure_file (HOOMDVersion.h.in ${HOOMD_BINARY_DIR}/hoomd/include/HOOMDVersion.h)

install(FILES ${HOOMD_BINARY_DIR}/hoomd/include/HOOMDVersion.h
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include
        )

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
                   GetarDumpWriter.cc
                   GetarInitializer.cc
                   GSDDumpWriter.cc
                   GSDReader.cc
                   HOOMDMath.cc
                   HOOMDVersion.cc
                   IMDInterface.cc
                   Initializers.cc
                   Integrator.cc
                   IntegratorData.cc
                   LoadBalancer.cc
                   Logger.cc
                   LogPlainTXT.cc
                   LogMatrix.cc
                   LogHDF5.cc
                   Messenger.cc
                   MemoryTraceback.cc
                   MPIConfiguration.cc
                   ParticleData.cc
                   ParticleGroup.cc
                   Profiler.cc
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
                   extern/vmdsock.cc
                   )

set(_hoomd_headers
    AABB.h
    AABBTree.h
    Analyzer.h
    Autotuner.h
    BondedGroupData.cuh
    BondedGroupData.h
    BoxDim.h
    BoxResizeUpdater.h
    CachedAllocator.h
    CallbackAnalyzer.h
    CellListGPU.cuh
    CellListGPU.h
    CellList.h
    CellListStencil.h
    ClockSource.h
    CommunicatorGPU.cuh
    CommunicatorGPU.h
    Communicator.h
    Compute.h
    ComputeThermoGPU.cuh
    ComputeThermoGPU.h
    ComputeThermo.h
    ComputeThermoTypes.h
    ConstForceCompute.h
    DCDDumpWriter.h
    DomainDecomposition.h
    ExecutionConfiguration.h
    Filesystem.h
    ForceCompute.h
    ForceConstraint.h
    GetarDumpIterators.h
    GetarDumpWriter.h
    GetarInitializer.h
    GlobalArray.h
    GPUArray.h
    GPUFlags.h
    GPUPartition.cuh
    GPUPolymorph.h
    GPUPolymorph.cuh
    GPUVector.h
    GSDDumpWriter.h
    GSDReader.h
    HalfStepHook.h
    HOOMDMath.h
    HOOMDMPI.h
    IMDInterface.h
    Index1D.h
    Initializers.h
    Integrator.cuh
    IntegratorData.h
    Integrator.h
    LoadBalancerGPU.cuh
    LoadBalancerGPU.h
    LoadBalancer.h
    Logger.h
    LogPlainTXT.h
    LogMatrix.h
    LogHDF5.h
    managed_allocator.h
    ManagedArray.h
    MemoryTraceback.h
    Messenger.h
    MPIConfiguration.h
    ParticleData.cuh
    ParticleData.h
    ParticleGroup.cuh
    ParticleGroup.h
    Profiler.h
    RandomNumbers.h
    RNGIdentifiers.h
    Saru.h
    SFCPackUpdaterGPU.cuh
    SFCPackUpdaterGPU.h
    SFCPackUpdater.h
    SharedSignal.h
    SignalHandler.h
    SnapshotSystemData.h
    SystemDefinition.h
    System.h
    TextureTools.h
    Updater.h
    Variant.h
    VectorMath.h
    WarpTools.cuh
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

# include libgetar sources directly into _hoomd.so
get_property(GETAR_SRCS_REL TARGET getar PROPERTY SOURCES)
foreach(getar_src ${GETAR_SRCS_REL})
  list(APPEND _hoomd_sources ${libgetar_DIR}/${getar_src})
endforeach(getar_src)

# add libgetar-specific defines
get_property(HOOMD_DEFS DIRECTORY . PROPERTY COMPILE_DEFINITIONS)
get_property(GETAR_DEFS DIRECTORY ${libgetar_DIR} PROPERTY COMPILE_DEFINITIONS)
list(REMOVE_ITEM GETAR_DEFS ${HOOMD_DEFS})
foreach(def ${GETAR_DEFS})
  add_definitions(-D${def})
endforeach(def)

# add libgetar-specific libraries
get_property(GETAR_LIBS TARGET getar PROPERTY LINK_LIBRARIES)
list(APPEND HOOMD_COMMON_LIBS ${GETAR_LIBS})

# quiet some warnings locally on files we can't modify
if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set_source_files_properties("extern/libgetar/vogl_miniz/vogl_miniz.cpp" PROPERTIES COMPILE_FLAGS "-Wno-strict-aliasing")
endif()

# add neighbor library
if (ENABLE_CUDA)
    add_subdirectory(neighbor)
    foreach(cxx_src ${NEIGHBOR_CXX_SRCS})
        list(APPEND _hoomd_sources neighbor/${cxx_src})
    endforeach(cxx_src)
    foreach(cu_src ${NEIGHBOR_CU_SRCS})
        list(APPEND _hoomd_cu_sources neighbor/${cu_src})
    endforeach(cu_src)
    foreach(header ${NEIGHBOR_HEADERS})
        list(APPEND _hoomd_headers neighbor/${header})
    endforeach(header)
endif (ENABLE_CUDA)

if (ENABLE_CUDA)
CUDA_COMPILE(_CUDA_GENERATED_FILES ${_hoomd_cu_sources} OPTIONS ${CUDA_ADDITIONAL_OPTIONS} SHARED)
endif (ENABLE_CUDA)

pybind11_add_module(_hoomd SHARED module.cc ${_hoomd_sources} ${_CUDA_GENERATED_FILES} NO_EXTRAS)

# add quick hull as its own library so that it's symbols can be public
add_library (quickhull SHARED extern/quickhull/QuickHull.cpp)

# link the library to its dependencies
target_link_libraries(_hoomd PUBLIC quickhull)
target_link_libraries(_hoomd PRIVATE ${HOOMD_COMMON_LIBS})
if (APPLE)
set_target_properties(_hoomd PROPERTIES INSTALL_RPATH "@loader_path")
else()
set_target_properties(_hoomd PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

set(HOOMD_LIBRARIES _hoomd quickhull ${HOOMD_COMMON_LIBS})

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

fix_cudart_rpath(_hoomd)

# install the library
install(TARGETS _hoomd
        LIBRARY DESTINATION ${PYTHON_MODULE_BASE_DIR}
        )
install(TARGETS quickhull
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
          update.py
          util.py
          variant.py
          lattice.py
          hdf5.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

# copy python modules to the build directory to make it a working python package
MACRO(copy_header file)
    add_custom_command (
        OUTPUT ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/include/hoomd/${file}
        COMMENT    "Copy hoomd/include/hoomd/${file}"
    )
ENDMACRO(copy_header)

add_custom_target(copy_hoomd ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_hoomd_headers}
        DESTINATION ${PYTHON_MODULE_BASE_DIR}/include/hoomd
       )

if (COPY_HEADERS)
# copy headers into build directory
foreach(file ${_hoomd_headers})
    copy_header(${file})
endforeach()

add_custom_target(copy_hoomd_headers ALL DEPENDS ${_hoomd_headers})
endif()

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
    if (ENABLE_CUDA)
        add_subdirectory(neighbor/test)
    endif (ENABLE_CUDA)
endif()

if (BUILD_MD)
    if (ENABLE_MPI)
        # add the distributed FFT library
        add_subdirectory(${HOOMD_SOURCE_DIR}/hoomd/extern/dfftlib)
    endif()

    add_subdirectory(md)
endif()

if (NOT SINGLE_PRECISION AND BUILD_HPMC)
    add_subdirectory(hpmc)
endif()

if (BUILD_DEPRECATED)
    add_subdirectory(deprecated)
endif()

if (BUILD_CGCMM AND BUILD_MD)
    add_subdirectory(cgcmm)
endif()

if (BUILD_METAL AND BUILD_MD)
    add_subdirectory(metal)
endif()

if (BUILD_DEM AND BUILD_MD)
    add_subdirectory(dem)
endif()

if (BUILD_MPCD)
    add_subdirectory(mpcd)
endif()

if (BUILD_JIT)
    add_subdirectory(jit)
endif()

file(GLOB _directory_contents RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

# explicitly remove packages which are already explicitly dealt with
list(REMOVE_ITEM _directory_contents test test-py extern md hpmc deprecated cgcmm metal dem mpcd neighbor jit)

foreach(entry ${_directory_contents})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${entry} OR IS_SYMLINK ${CMAKE_CURRENT_SOURCE_DIR}/${entry})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${entry}/CMakeLists.txt)
            add_subdirectory(${entry})
        endif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${entry}/CMakeLists.txt)
    endif(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${entry} OR IS_SYMLINK ${CMAKE_CURRENT_SOURCE_DIR}/${entry})
endforeach()
