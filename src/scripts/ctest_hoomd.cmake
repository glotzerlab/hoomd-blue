# ctest -S script for testing HOOMD and submitting to the dashboard at my.cdash.org
# this script must be copied and modified for each test build. Locations in the script
# that need to be modified are labeled **MODIFY THIS and are mostly grouped at the beginning

# instructions on use:
# 1) checkout a copy of hoomd's source to be tested
# 2) make a bin/ directory somewhere
# 3) modify the variables in this script to match the desired test parameters
# 4) set TEST_GROUP to "Experimental" and run ctest -V -S ctest_hoomd.cmake to check that the test runs
#     the results of the test should show up at: http://my.cdash.org/index.php?project=HOOMD
#     (you may want to ignore the bdnvt and npt tests for this as they are quite long)
# 5) chate TEST_GROUP back to "Nightly" and set "ctest -S ctest_hoomd.cmake"  to run every day

# **MODIFY THIS (location where hoomd src and bin directory are)
SET (CTEST_SOURCE_DIRECTORY "$ENV{HOME}/hoomd/src")
SET (CTEST_BINARY_DIRECTORY "$ENV{HOME}/hoomd/bin_ctest")

# **MODIFY THIS (for testing the test)
SET (TEST_GROUP "Experimental")
# SET (TEST_GROUP "Nightly")

# **MODIFY THIS (name of computer performing the tests)
SET (SITE_NAME "sitename")

# **MODIFY THIS (name of hoomd branch you are testing)
#SET (HOOMD_BRANCH "trunk")
SET (HOOMD_BRANCH "hoomd-0.8")

# **MODIFY THIS (name of the system)
set (SYSTEM_NAME "Linux")

# **MODIFY THIS (a string identifying the compiler: this cannot be autodetected yet)
SET (COMPILER_NAME "gcc412")

# **MODIFY THIS (set to ON to enable CUDA build)
SET (ENABLE_CUDA "OFF")

# **MODIFY THIS (set to OFF to enable double precision build) (ENABLE_CUDA must be off if this is set off)
SET (SINGLE_PRECISION "ON")

# **MODIFY THIS (set to OFF to enable shared library builds)
SET (ENABLE_STATIC "ON")

# **MODIFY THIS (set tests to ignore, see the example for the format) (bdnvt and npt take minutes to run, and an enternity with valgrind enabled)
SET (IGNORE_TESTS "")
#SET (IGNORE_TESTS "-E \"bdnvt|npt\"")

# other stuff
SET (CTEST_SVN_COMMAND "svn")
SET (CTEST_COMMAND "ctest -D ${TEST_GROUP} ${IGNORE_TESTS}")
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

if (SINLE_PRECISION MATCHES "ON")
	SET (BUILDNAME "${BUILDNAME}-single")
else (SINLE_PRECISION MATCHES "ON")
	SET (BUILDNAME "${BUILDNAME}-double")
endif (SINLE_PRECISION MATCHES "ON")

if (ENABLE_CUDA MATCHES "ON")
	SET (BUILDNAME "${BUILDNAME}-cuda")
endif (ENABLE_CUDA MATCHES "ON")

SET (CTEST_INITIAL_CACHE "
CMAKE_GENERATOR:INTERNAL=Unix Makefiles
BUILDNAME:STRING=${BUILDNAME}
SITE:STRING=${SITE_NAME}
CMAKE_BUILD_TYPE:STRING=Debug
ENABLE_CUDA:BOOL=${ENABLE_CUDA}
SINGLE_PRECISION:BOOL=${SINGLE_PRECISION}
ENABLE_STATIC:BOOL=${ENABLE_STATIC}
")
