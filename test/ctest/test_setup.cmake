## Don't edit this file unless you really know what you are doing
# See ctest_hoomd.cmake for complete documentation

# lets create a build name to idenfity all these options in a string. It will look like
# Linux-gcc412-trunk-single-cuda (for a single precision build with cuda)
# Linux-gcc412-hoomd-0.8-double (for a double precision build without cuda and in the hoomd-0.8 branch)
SET (BUILDNAME "${SYSTEM_NAME}")

if (NOT HOOMD_BRANCH MATCHES "master")
    SET (BUILDNAME "${BUILDNAME}-${HOOMD_BRANCH}")
endif ()

if (ENABLE_STATIC MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-static")
endif ()

if (NOT SINGLE_PRECISION MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-double")
else ()
    SET (BUILDNAME "${BUILDNAME}-single")
endif ()

if (NOT ENABLE_CUDA MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-cpu")
else ()
    SET (BUILDNAME "${BUILDNAME}-cuda")
endif ()

if (NOT BUILD_TYPE MATCHES "Release")
    SET (BUILDNAME "${BUILDNAME}-${BUILD_TYPE}")
endif ()

if (ENABLE_MPI MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-mpi")
endif ()

if (GPU_GENERATION)
    SET (BUILDNAME "${BUILDNAME}-${GPU_GENERATION}")
endif ()

if (ENABLE_COVERAGE)
    SET (COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage")
endif (ENABLE_COVERAGE)

SET (CTEST_COMMAND "ctest -D ${TEST_GROUP} ${IGNORE_TESTS}")
if (MEMORYCHECK_COMMAND)
    set (CTEST_COMMAND "${CTEST_COMMAND}"
            "ctest -D ${TEST_GROUP}MemCheck -D ${TEST_GROUP}Submit ${IGNORE_TESTS}")
endif (MEMORYCHECK_COMMAND)

SET (CTEST_INITIAL_CACHE "
CMAKE_GENERATOR:INTERNAL=Unix Makefiles
MAKECOMMAND:STRING=/usr/bin/make -i ${MAKEOPTIONS}
BUILDNAME:STRING=${BUILDNAME}
SITE:STRING=${SITE_NAME}
CMAKE_BUILD_TYPE:STRING=${BUILD_TYPE}
ENABLE_CUDA:BOOL=${ENABLE_CUDA}
ENABLE_DOXYGEN:BOOL=OFF
ENABLE_MPI:BOOL=${ENABLE_MPI}
SINGLE_PRECISION:BOOL=${SINGLE_PRECISION}
ENABLE_STATIC:BOOL=${ENABLE_STATIC}
ENABLE_TEST_ALL:BOOL=ON
CMAKE_C_FLAGS:STRING=${COVERAGE_FLAGS}
CMAKE_CXX_FLAGS:STRING=${COVERAGE_FLAGS}
MEMORYCHECK_COMMAND:FILEPATH=${MEMORYCHECK_COMMAND}
MEMORYCHECK_SUPPRESSIONS_FILE:FILEPATH=${CTEST_CHECKOUT_DIR}/test/unit/combined_valgrind.supp
CUDA_ARCH_LIST:STRING=${CUDA_ARCH_LIST}
")
