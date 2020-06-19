add_subdirectory(extern)

# configure the version information file
configure_file (HOOMDVersion.h.in ${HOOMD_BINARY_DIR}/hoomd/include/HOOMDVersion.h)

install(FILES ${HOOMD_BINARY_DIR}/hoomd/include/HOOMDVersion.h
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include
        )

##############################
## Source setup

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
                   ComputeThermoHMA.cc
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
    ComputeThermoHMAGPU.cuh
    ComputeThermoHMAGPU.h
    ComputeThermo.h
    ComputeThermoHMA.h
    ComputeThermoTypes.h
    ComputeThermoHMATypes.h
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
    GSDShapeSpecWriter.h
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

if (ENABLE_HIP)
list(APPEND _hoomd_sources CellListGPU.cc
                           CommunicatorGPU.cc
                           ComputeThermoGPU.cc
                           ComputeThermoHMAGPU.cc
                           LoadBalancerGPU.cc
                           SFCPackUpdaterGPU.cc
                           )
endif()

set(_hoomd_cu_sources BondedGroupData.cu
                      CellListGPU.cu
                      CommunicatorGPU.cu
                      ComputeThermoGPU.cu
                      ComputeThermoHMAGPU.cu
                      Integrator.cu
                      LoadBalancerGPU.cu
                      ParticleData.cu
                      ParticleGroup.cu
                      SFCPackUpdaterGPU.cu)

# include libgetar sources directly into _hoomd.so
get_property(GETAR_SRCS_REL TARGET getar PROPERTY SOURCES)
foreach(getar_src ${GETAR_SRCS_REL})
  list(APPEND _hoomd_sources ${libgetar_DIR}/${getar_src})
endforeach(getar_src)

# quiet some warnings locally on files we can't modify
if(CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set_source_files_properties("extern/libgetar/vogl_miniz/vogl_miniz.cpp" PROPERTIES COMPILE_FLAGS "-Wno-strict-aliasing")
endif()

if (ENABLE_HIP)
set(_cuda_sources ${_hoomd_cu_sources})
endif (ENABLE_HIP)

#########################
## Build the module
pybind11_add_module(_hoomd SHARED module.cc ${_hoomd_sources} ${_cuda_sources} ${_hoomd_headers} NO_EXTRAS)

# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_hoomd ALIAS _hoomd)

# add quick hull as its own library so that it's symbols can be public
add_library (quickhull SHARED extern/quickhull/QuickHull.cpp)

# link the library to its dependencies
target_link_libraries(_hoomd PUBLIC pybind11::pybind11 quickhull Eigen3::Eigen)

# specify required include directories
target_include_directories(_hoomd PUBLIC
                                  $<BUILD_INTERFACE:${HOOMD_SOURCE_DIR}>
                                  $<INSTALL_INTERFACE:${PYTHON_SITE_INSTALL_DIR}/include>)

target_include_directories(_hoomd PUBLIC
                                  $<BUILD_INTERFACE:${HOOMD_BINARY_DIR}/hoomd/include>)

# specify required definitions
target_compile_definitions(_hoomd PUBLIC _REENTRANT EIGEN_MPL2_ONLY)

if (SINGLE_PRECISION)
    target_compile_definitions(_hoomd PUBLIC SINGLE_PRECISION)
endif(SINGLE_PRECISION)

# Libraries and compile definitions for CUDA enabled builds
if (ENABLE_HIP)
    if (HIP_PLATFORM STREQUAL "hcc")
        target_link_libraries(_hoomd PUBLIC HIP::hipfft HIP::hiprt)
    elseif(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries(_hoomd PUBLIC CUDA::cudart CUDA::cufft)
    endif()
    target_compile_definitions(_hoomd PUBLIC ENABLE_HIP CUDA_ARCH=${_cuda_min_arch})

    if(ALWAYS_USE_MANAGED_MEMORY)
        target_compile_definitions(_hoomd PUBLIC ALWAYS_USE_MANAGED_MEMORY)
    endif()

    if (ENABLE_NVTOOLS)
        target_link_libraries(_hoomd PUBLIC CUDA::nvToolsExt)
        target_compile_definitions(_hoomd PUBLIC ENABLE_NVTOOLS)
    endif()

    if (CUSOLVER_AVAILABLE)
        target_compile_definitions(_hoomd PUBLIC CUSOLVER_AVAILABLE)
    endif()

    target_link_libraries(_hoomd PUBLIC HIP::hip)

    if (ENABLE_ROCTRACER)
        target_link_libraries(_hoomd PUBLIC HIP::roctracer)
        target_compile_definitions(_hoomd PUBLIC ENABLE_ROCTRACER)
    endif()
endif()


# Libraries and compile definitions for TBB enabled builds
if (ENABLE_TBB)
    find_package_config_first(TBB 4.3)

    if (TBB_FOUND)
        get_target_property(_tbb_library TBB::tbb IMPORTED_LOCATION_RELEASE)
        get_target_property(_tbb_include_dir TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
        find_package_message(tbb "Found TBB: ${TBB_DIR} ${_tbb_library} ${_tbb_include_dir}" "[${_tbb_library}][${_tbb_include_dir}]")
    endif()

    target_compile_definitions(_hoomd PUBLIC ENABLE_TBB)
    target_link_libraries(_hoomd PUBLIC TBB::tbb)
endif()

# Libraries and compile definitions for MPI enabled builds
if (ENABLE_MPI)
    target_compile_definitions(_hoomd PUBLIC ENABLE_MPI)
    target_link_libraries(_hoomd PUBLIC MPI::MPI_CXX cereal)

    if (ENABLE_MPI_CUDA)
          target_compile_definitions(_hoomd PUBLIC ENABLE_MPI_CUDA)
    endif(ENABLE_MPI_CUDA)
endif()

if (ENABLE_HPMC_MIXED_PRECISION)
    target_compile_definitions(_hoomd PUBLIC ENABLE_HPMC_MIXED_PRECISION)
endif()

if (APPLE)
set_target_properties(_hoomd PROPERTIES INSTALL_RPATH "@loader_path")
else()
set_target_properties(_hoomd PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

fix_cudart_rpath(_hoomd)

# install the library
install(TARGETS _hoomd quickhull EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}
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
          device.py
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
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_hoomd ALL DEPENDS ${files})

# install headers in installation target
install(FILES ${_hoomd_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd
       )

if (BUILD_TESTING)
    add_subdirectory(test-py)
    add_subdirectory(test)
endif()

##################################################
## Build components

if (BUILD_MD)
    if (ENABLE_MPI)
        # add the distributed FFT library
        add_subdirectory(${HOOMD_SOURCE_DIR}/hoomd/extern/dfftlib)
    endif()

    if (ENABLE_HIP)
        # add the neighbor library
        set(NEIGHBOR_HOOMD_INTERNAL TRUE)

        if (HIP_PLATFORM STREQUAL "hcc")
            set(NEIGHBOR_DISABLE_MIXED TRUE)
            target_compile_definitions(_hoomd PUBLIC NEIGHBOR_DISABLE_MIXED)
        endif()

        add_subdirectory(${HOOMD_SOURCE_DIR}/hoomd/extern/neighbor)
    endif (ENABLE_HIP)

    add_subdirectory(md)
endif()

if (NOT SINGLE_PRECISION AND BUILD_HPMC)
    add_subdirectory(hpmc)
endif()

if (BUILD_METAL AND BUILD_MD)
    add_subdirectory(metal)
endif()

if (BUILD_DEM AND BUILD_MD)
    add_subdirectory(dem)
endif()

if (NOT ENABLE_HIP OR HIP_PLATFORM STREQUAL "nvcc")
    if (BUILD_MPCD)
        add_subdirectory(mpcd)
    endif()
endif()

if (BUILD_JIT)
    add_subdirectory(jit)
endif()

file(GLOB _directory_contents RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

# explicitly remove packages which are already explicitly dealt with
list(REMOVE_ITEM _directory_contents test test-py extern md hpmc metal dem mpcd jit)

foreach(entry ${_directory_contents})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${entry} OR IS_SYMLINK ${CMAKE_CURRENT_SOURCE_DIR}/${entry})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${entry}/CMakeLists.txt)
            add_subdirectory(${entry})
        endif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${entry}/CMakeLists.txt)
    endif(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${entry} OR IS_SYMLINK ${CMAKE_CURRENT_SOURCE_DIR}/${entry})
endforeach()
