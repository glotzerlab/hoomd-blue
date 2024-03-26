add_subdirectory(extern)

# configure the version information file
configure_file (HOOMDVersion.h.inc ${HOOMD_BINARY_DIR}/hoomd/include/HOOMDVersion.h)

install(FILES ${HOOMD_BINARY_DIR}/hoomd/include/HOOMDVersion.h
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include
        )

# translate cmake true/false to Python
if (BUILD_MD)
    set(_md_built "True")
else()
    set(_md_built "False")
endif()

if (BUILD_MD)
    set(_md_built "True")
else()
    set(_md_built "False")
endif()

if (BUILD_HPMC)
    set(_hpmc_built "True")
else()
    set(_hpmc_built "False")
endif()

if (BUILD_METAL)
    set(_metal_built "True")
else()
    set(_metal_built "False")
endif()

if (BUILD_MPCD)
    set(_mpcd_built "True")
else()
    set(_mpcd_built "False")
endif()

if (ENABLE_LLVM)
    set(_llvm_enabled "True")
else()
    set(_llvm_enabled "False")
endif()

configure_file (version_config.py.in ${HOOMD_BINARY_DIR}/hoomd/version_config.py)
install(FILES ${HOOMD_BINARY_DIR}/hoomd/version_config.py
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}
        )

##############################
## Source setup

set(_hoomd_sources Action.cc
                   Autotuned.cc
                   Analyzer.cc
                   BondedGroupData.cc
                   BoxResizeUpdater.cc
                   CellList.cc
                   CellListStencil.cc
                   ClockSource.cc
                   Communicator.cc
                   CommunicatorGPU.cc
                   Compute.cc
                   DCDDumpWriter.cc
                   DomainDecomposition.cc
                   ExecutionConfiguration.cc
                   ForceCompute.cc
                   ForceConstraint.cc
                   GSDDequeWriter.cc
                   GSDDumpWriter.cc
                   GSDReader.cc
                   HOOMDMath.cc
                   HOOMDVersion.cc
                   Initializers.cc
                   Integrator.cc
                   LoadBalancer.cc
                   MeshGroupData.cc
                   MeshDefinition.cc
                   Messenger.cc
                   MPIConfiguration.cc
                   ParticleData.cc
                   ParticleGroup.cc
                   ParticleFilterUpdater.cc
                   PythonLocalDataAccess.cc
                   PythonAnalyzer.cc
                   PythonTuner.cc
                   PythonUpdater.cc
                   SFCPackTuner.cc
                   SnapshotSystemData.cc
                   System.cc
                   SystemDefinition.cc
                   Trigger.cc
                   Tuner.cc
                   Updater.cc
                   Variant.cc
                   VectorVariant.cc
                   extern/BVLSSolver.cc
                   extern/gsd.c
                   extern/kiss_fft.cc
                   extern/kiss_fftnd.cc
                   filter/export_filters.cc
                   )

# ignore conversion warnings in external files
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(extern/kiss_fft.cc PROPERTIES COMPILE_FLAGS "-Wno-conversion -Wno-float-conversion")
endif()

set(_hoomd_headers
    AABB.h
    AABBTree.h
    Action.h
    Analyzer.h
    ArrayView.h
    Autotuned.h
    Autotuner.h
    BondedGroupData.cuh
    BondedGroupData.h
    BoxDim.h
    BoxResizeUpdater.h
    BoxResizeUpdaterGPU.cuh
    BoxResizeUpdaterGPU.h
    UpdaterRemoveDrift.h
    CachedAllocator.h
    CellListGPU.cuh
    CellListGPU.h
    CellList.h
    CellListStencil.h
    ClockSource.h
    CommunicatorGPU.cuh
    CommunicatorGPU.h
    Communicator.h
    Compute.h
    DCDDumpWriter.h
    DomainDecomposition.h
    ExecutionConfiguration.h
    Filesystem.h
    ForceCompute.h
    ForceConstraint.h
    GlobalArray.h
    GPUArray.h
    GPUFlags.h
    GPUPartition.cuh
    GPUVector.h
    GSD.h
    GSDDequeWriter.h
    GSDDumpWriter.h
    GSDReader.h
    HalfStepHook.h
    HOOMDMath.h
    HOOMDMPI.h
    Index1D.h
    Initializers.h
    Integrator.cuh
    Integrator.h
    LoadBalancerGPU.cuh
    LoadBalancerGPU.h
    LoadBalancer.h
    managed_allocator.h
    ManagedArray.h
    MeshGroupData.h
    MeshDefinition.h
    Messenger.h
    MPIConfiguration.h
    ParticleData.cuh
    ParticleData.h
    ParticleGroup.cuh
    ParticleGroup.h
    ParticleFilterUpdater.h
    PythonLocalDataAccess.h
    PythonUpdater.h
    PythonAnalyzer.h
    RandomNumbers.h
    RNGIdentifiers.h
    SFCPackTunerGPU.cuh
    SFCPackTunerGPU.h
    SFCPackTuner.h
    SharedSignal.h
    SnapshotSystemData.h
    SystemDefinition.h
    System.h
    Trigger.h
    Tuner.h
    TextureTools.h
    Updater.h
    Variant.h
    VectorVariant.h
    VectorMath.h
    WarpTools.cuh
    )

if (ENABLE_HIP)
list(APPEND _hoomd_sources BoxResizeUpdaterGPU.cc
                           CellListGPU.cc
                           CommunicatorGPU.cc
                           LoadBalancerGPU.cc
                           SFCPackTunerGPU.cc
                           )
endif()

set(_hoomd_cu_sources BondedGroupData.cu
                      BoxResizeUpdaterGPU.cu
                      CellListGPU.cu
                      CommunicatorGPU.cu
                      Integrator.cu
                      LoadBalancerGPU.cu
                      ParticleData.cu
                      ParticleGroup.cu
                      SFCPackTunerGPU.cu)

# add the MPCD base parts that should go into _hoomd (i.e., core particle data)
if (BUILD_MPCD)
    list(APPEND _hoomd_sources
        mpcd/ParticleData.cc
        mpcd/ParticleDataSnapshot.cc
        )
    list(APPEND _hoomd_cu_sources
        mpcd/ParticleData.cu
    )
    # headers will be installed by MPCD itself so they go in the right subdirectory
endif()

if (ENABLE_HIP)
set(_cuda_sources ${_hoomd_cu_sources})
set_source_files_properties(${_hoomd_cu_sources} PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE})
endif (ENABLE_HIP)

#########################
## Build the module
hoomd_add_module(_hoomd SHARED module.cc ${_hoomd_sources} ${_cuda_sources} ${_hoomd_headers} NO_EXTRAS)

# alias into the HOOMD namespace so that plugins and symlinked components both work
add_library(HOOMD::_hoomd ALIAS _hoomd)

# Work around support for the delete operator with pybind11 and older versions of clang
# https://github.com/pybind/pybind11/issues/1604
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(_hoomd PUBLIC $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<STREQUAL:${HIP_PLATFORM},nvcc>>:-Xcompiler=>;-fsized-deallocation)
endif()

# add quick hull as its own library so that it's symbols can be public
add_library (quickhull SHARED extern/quickhull/QuickHull.cpp)

# ignore conversion warnings in quickull
if(CMAKE_COMPILER_IS_GNUCXX)
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0.0)
        target_compile_options(quickhull PRIVATE "-Wno-conversion;-Wno-pessimizing-move")
    else()
        target_compile_options(quickhull PRIVATE "-Wno-conversion")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(quickhull PRIVATE "-Wno-conversion;-Wno-pessimizing-move")
endif()

# link the library to its dependencies
target_link_libraries(_hoomd PUBLIC pybind11::pybind11 quickhull Eigen3::Eigen)
if(CMAKE_SYSTEM_NAME STREQUAL "FreeBSD")
    target_link_libraries(_hoomd PUBLIC execinfo) # on FreeBSD backtrace() is in libexecinfo
endif()

# specify required include directories
target_include_directories(_hoomd PUBLIC
                                  $<BUILD_INTERFACE:${HOOMD_SOURCE_DIR}>
                                  $<INSTALL_INTERFACE:${PYTHON_SITE_INSTALL_DIR}/include>)

target_include_directories(_hoomd PUBLIC
                                  $<BUILD_INTERFACE:${HOOMD_BINARY_DIR}/hoomd/include>)

# specify required definitions
target_compile_definitions(_hoomd PUBLIC _REENTRANT EIGEN_MPL2_ONLY)
target_compile_definitions(_hoomd PUBLIC HOOMD_SHORTREAL_SIZE=${HOOMD_SHORTREAL_SIZE})
target_compile_definitions(_hoomd PUBLIC HOOMD_LONGREAL_SIZE=${HOOMD_LONGREAL_SIZE})

# Libraries and compile definitions for CUDA enabled builds
if (ENABLE_HIP)
    if (HIP_PLATFORM STREQUAL "hip-clang")
        target_link_libraries(_hoomd PUBLIC hip::hipfft)
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

    target_link_libraries(_hoomd PUBLIC hip::host)

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
    target_link_libraries(_hoomd PUBLIC MPI::MPI_CXX cereal::cereal)
endif()

if (APPLE)
set_target_properties(_hoomd PROPERTIES INSTALL_RPATH "@loader_path")
else()
set_target_properties(_hoomd PROPERTIES INSTALL_RPATH "\$ORIGIN")
endif()

# install the library
install(TARGETS _hoomd quickhull EXPORT HOOMDTargets
        LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}
        )

################ Python only modules
# copy python modules to the build directory to make it a working python package
set(files box.py
          communicator.py
          _compile.py
          conftest.py
          device.py
          __init__.py
          error.py
          operation.py
          operations.py
          pytest_plugin_validate.py
          util.py
          simulation.py
          state.py
          trigger.py
          snapshot.py
          logging.py
          mesh.py
          version.py
          wall.py
          pytest.ini
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}
       )

# version_config.py is generated by configure_file, ignore here
copy_files_to_build("${files}" "hoomd" "*.py" "version_config.py")

# install headers in installation target
install(FILES ${_hoomd_headers}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd
       )

# subdirectories that are not components
add_subdirectory(custom)
add_subdirectory(data)
add_subdirectory(filter)
add_subdirectory(write)
add_subdirectory(pytest)
add_subdirectory(tune)
add_subdirectory(update)
add_subdirectory(variant)

if (BUILD_TESTING)
    # add_subdirectory(test-py)
    add_subdirectory(test)
endif()

##################################################
## Build components

if (BUILD_MD)
    if (ENABLE_MPI)
        # add the distributed FFT library
        add_subdirectory(${HOOMD_SOURCE_DIR}/hoomd/extern/dfftlib)
    endif()
    add_subdirectory(md)
endif()

if (BUILD_HPMC)
    add_subdirectory(hpmc)
endif()

if (BUILD_METAL AND BUILD_MD)
    add_subdirectory(metal)
endif()

if (BUILD_MPCD)
    target_compile_definitions(_hoomd PUBLIC BUILD_MPCD)
    add_subdirectory(mpcd)
endif()

foreach(entry ${PLUGINS})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${entry}/CMakeLists.txt)
        message(STATUS "Found plugin: ${entry}")
        add_subdirectory(${entry})
    endif()
endforeach()
