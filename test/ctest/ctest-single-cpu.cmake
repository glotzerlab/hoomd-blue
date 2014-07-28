# ctest -S script for testing HOOMD and submitting to the dashboard at cdash.fourpisolutions.com
# this script must be copied and modified for each test build. Locations in the script
# that need to be modified to configure the build are near the top

# instructions on use:
# 1) checkout a copy of hoomd's source to be tested
# 2) copy all ctest_hoomd_* cmake scripts to a convenient location (i.e., next to the hoomd source checkout)
# 3a) On linux/mac: cp ctest_hoomd_setup_linux.cmake ctest_hoomd_setup.cmake
# 3b) On win32: cp ctest_hoomd_setup_win32.cmake ctest_hoomd_setup.cmake
# 3c) On win64: cp ctest_hoomd_setup_win64.cmake ctest_hoomd_setup.cmake
# 4) modify variables in ctest_site_options to match your site
# 5) set TEST_GROUP to "Experimental" and run ctest -V -S ctest_hoomd.cmake to check that the test runs OK.
#     Test results of the test should show up at: http://cdash.fourpisolutions.com/index.php?project=HOOMD.
#     (you may want to ignore the bdnvt and npt tests for this as they are quite long).
# 6) change TEST_GROUP back to "Nightly" and set "ctest -S ctest_hoomd.cmake" to run every day

# ctest_hoomd.cmake tests the default configuration. Also included are a set of of other scripts with
# various combinations of build options. Use any or all of them as you wish.

# (set to ON to enable CUDA build)
SET (ENABLE_CUDA "OFF")

# (set to OFF to enable double precision build) (ENABLE_CUDA must be off if this is set off)
SET (SINGLE_PRECISION "ON")

# (set to OFF to enable shared library builds)
SET (ENABLE_STATIC "OFF")

# (set to ON to enable MPI)
SET (ENABLE_MPI "OFF")

# (set tests to ignore, see the example for the format)
# (bdnvt and npt take minutes to run, and an enternity with valgrind enabled, so they are ignored by default)
SET (IGNORE_TESTS "")
#SET (IGNORE_TESTS "-E \"test_bdnvt_integrator|test_npt_integrator\"")

# (location of valgrind: Leave blank unless you REALLY want the long valgrind tests to run
SET (MEMORYCHECK_COMMAND "")
#SET (MEMORYCHECK_COMMAND "/usr/bin/valgrind")

# (architectures to compile CUDA for 10=compute 1.0 11=compute 1.1, ...)
SET (CUDA_ARCH_LIST 20 30 35)

# (set to ON to enable coverage tests: these extensive tests don't really need to be done on every single build)
SET (ENABLE_COVERAGE OFF)

# Build type
SET (BUILD_TYPE Release)

# Bring in the settings common to all ctest scripts
include(site_options.cmake)
include(test_setup.cmake)
