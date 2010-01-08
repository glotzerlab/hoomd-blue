## Don't edit this file unless you really know what you are doing
# See ctest_hoomd.cmake for complete documentation

# shared lib builds are not supported on windows, force static
set(ENABLE_STATIC "ON")

# lets create a build name to idenfity all these options in a string. It will look like
# Linux-gcc412-trunk-single-cuda (for a single precision build with cuda)
# Linux-gcc412-hoomd-0.8-double (for a double precision build without cuda and in the hoomd-0.8 branch)
SET (BUILDNAME "${SYSTEM_NAME}-${COMPILER_NAME}-${HOOMD_BRANCH}")

if (ENABLE_STATIC MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-static")
else(ENABLE_STATIC MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-shared")
endif(ENABLE_STATIC MATCHES "ON")

if (SINGLE_PRECISION MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-single")
else (SINGLE_PRECISION MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-double")
endif (SINGLE_PRECISION MATCHES "ON")

if (ENABLE_CUDA MATCHES "ON")
    SET (BUILDNAME "${BUILDNAME}-cuda")
    if (CUDA_BUILD_EMULATION MATCHES "ON")
        SET (BUILDNAME "${BUILDNAME}-emu")
    endif (CUDA_BUILD_EMULATION MATCHES "ON")
endif (ENABLE_CUDA MATCHES "ON")

set (CTEST_CONFIGURATION_TYPE "Release")
SET (CTEST_COMMAND "ctest -C ${CTEST_CONFIGURATION_TYPE} -D ${TEST_GROUP} ${IGNORE_TESTS}")

SET (CTEST_INITIAL_CACHE "
CMAKE_GENERATOR:INTERNAL=Visual Studio 8 2005
BUILDNAME:STRING=${BUILDNAME}
SITE:STRING=${SITE_NAME}
CMAKE_BUILD_TYPE:STRING=Debug
ENABLE_CUDA:BOOL=${ENABLE_CUDA}
SINGLE_PRECISION:BOOL=${SINGLE_PRECISION}
ENABLE_STATIC:BOOL=${ENABLE_STATIC}
ENABLE_TEST_ALL:BOOL="ON"
CMAKE_C_FLAGS:STRING=${COVERAGE_FLAGS}
CMAKE_CXX_FLAGS:STRING=${COVERAGE_FLAGS}
MEMORYCHECK_COMMAND:FILEPATH=${MEMORYCHECK_COMMAND}
MEMORYCHECK_SUPPRESSIONS_FILE:FILEPATH=${CTEST_CHECKOUT_DIR}/src/unit_tests/combined_valgrind.supp
CUDA_BUILD_EMULATION:STRING=${CUDA_BUILD_EMULATION}
CUDA_ARCH:STRING=${CUDA_ARCH}
")
