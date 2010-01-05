# ctest -S script for testing HOOMD and submitting to the dashboard at my.cdash.org
# this script must be copied and modified for each test build. Locations in the script
# that need to be modified to configure the build are near the top

# instructions on use:
# 1) checkout a copy of hoomd's source to be tested
# 2) make a bin/ directory somewhere
# 3) modify the variables in this script to match the desired test parameters
# 4) set TEST_GROUP to "Experimental" and run ctest -V -S ctest_hoomd.cmake to check that the test runs
#     the results of the test should show up at: http://cdash.fourpisolutions.com/index.php?project=HOOMD
#     (you may want to ignore the bdnvt and npt tests for this as they are quite long)
# 5) chate TEST_GROUP back to "Nightly" and set "ctest -S ctest_hoomd.cmake"  to run every day

# **MODIFY THESE 
# (location where hoomd src and bin directory are)
SET (CTEST_CHECKOUT_DIR "$ENV{HOME}/hoomd")
SET (CTEST_SOURCE_DIRECTORY "${CTEST_CHECKOUT_DIR}/src")
SET (CTEST_BINARY_DIRECTORY "${CTEST_CHECKOUT_DIR}/bin_ctest")
# (Experimental is for testing the test, Nightly is for production tests)
SET (TEST_GROUP "Experimental")  
# SET (TEST_GROUP "Nightly")
# (name of computer performing the tests)
SET (SITE_NAME "sitename")
# SET (SITE_NAME "rain.local")
# SET (SITE_NAME "photon.hopto.org")
# (name of hoomd branch you are testing)
#SET (HOOMD_BRANCH "trunk")
SET (HOOMD_BRANCH "hoomd-0.8")
# (name of the system)
set (SYSTEM_NAME "Linux")
# (a string identifying the compiler: this cannot be autodetected here)
SET (COMPILER_NAME "gcc412")
# (set to ON to enable CUDA build)
SET (ENABLE_CUDA "OFF")
# (set to OFF to enable double precision build) (ENABLE_CUDA must be off if this is set off)
SET (SINGLE_PRECISION "ON")
# (set to OFF to enable shared library builds)
SET (ENABLE_STATIC "ON")
# (set tests to ignore, see the example for the format) 
# (bdnvt and npt take minutes to run, and an enternity with valgrind enabled, so they are ignored by default)
# SET (IGNORE_TESTS "")
SET (IGNORE_TESTS "-E \"bdnvt|npt\"")
# (location of valgrind: Leave blank unless you REALLY want the long valgrind tests to run
SET (MEMORYCHECK_COMMAND "")
#SET (MEMORYCHECK_COMMAND "/usr/bin/valgrind")
# (change to emulation if you want to compile and test a GPU emulation build)
SET (CUDA_BUILD_EMULATION "OFF")
#SET (CUDA_BUILD_EMULATION "ON")
# (architecture to compile CUDA for 10=compute 1.0 11=compute 1.1, ...)
SET (CUDA_ARCH "10")
# (set to ON to enable coverage tests: these extensive tests don't really need to be done on every single build)
SET (ENABLE_COVERAGE OFF)


# other stuff that you might want to modify
SET (CTEST_SVN_COMMAND "svn")
SET (CTEST_COMMAND "ctest -D ${TEST_GROUP} ${IGNORE_TESTS}")
if (MEMORYCHECK_COMMAND)
	set (CTEST_COMMAND "${CTEST_COMMAND}" 
			"ctest -D ${TEST_GROUP}MemCheck -D ${TEST_GROUP}Submit ${IGNORE_TESTS}")
endif (MEMORYCHECK_COMMAND)
SET (CTEST_CMAKE_COMMAND "cmake")
SET (CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE)

#### Don't modify anything below unless you really know what you are doing

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

if (ENABLE_COVERAGE)
	SET (COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage")
endif (ENABLE_COVERAGE)


SET (CTEST_INITIAL_CACHE "
CMAKE_GENERATOR:INTERNAL=Unix Makefiles
MAKECOMMAND:STRING=/usr/bin/make -i -j 6
BUILDNAME:STRING=${BUILDNAME}
SITE:STRING=${SITE_NAME}
CMAKE_BUILD_TYPE:STRING=Debug
ENABLE_CUDA:BOOL=${ENABLE_CUDA}
SINGLE_PRECISION:BOOL=${SINGLE_PRECISION}
ENABLE_STATIC:BOOL=${ENABLE_STATIC}
CMAKE_C_FLAGS:STRING=${COVERAGE_FLAGS}
CMAKE_CXX_FLAGS:STRING=${COVERAGE_FLAGS}
MEMORYCHECK_COMMAND:FILEPATH=${MEMORYCHECK_COMMAND}
MEMORYCHECK_SUPPRESSIONS_FILE:FILEPATH=${CTEST_CHECKOUT_DIR}/src/unit_tests/combined_valgrind.supp
CUDA_BUILD_EMULATION:STRING=${CUDA_BUILD_EMULATION}
CUDA_ARCH:STRING=${CUDA_ARCH}
")
