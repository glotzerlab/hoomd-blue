# see ctest_hoomd.cmake for overall documentation
# this version of it tests a configuration different from the default

# (set to ON to enable CUDA build)
SET (ENABLE_CUDA "OFF")

# (set to ON to enable OpenMP build)
SET (ENABLE_OPENMP "ON")

# (set to OFF to enable double precision build) (ENABLE_CUDA must be off if this is set off)
SET (SINGLE_PRECISION "ON")

# (set to OFF to enable shared library builds)
SET (ENABLE_STATIC "OFF")

# (set tests to ignore, see the example for the format)
# (bdnvt and npt take minutes to run, and an enternity with valgrind enabled, so they are ignored by default)
SET (IGNORE_TESTS "")
#SET (IGNORE_TESTS "-E \"test_bdnvt_integrator|test_npt_integrator|test_npt_mtk_integrator\"")

# (location of valgrind: Leave blank unless you REALLY want the long valgrind tests to run
SET (MEMORYCHECK_COMMAND "")
#SET (MEMORYCHECK_COMMAND "/usr/bin/valgrind")

# (architectures to compile CUDA for 10=compute 1.0 11=compute 1.1, ...)
SET (CUDA_ARCH_LIST 12 13 20)

# (set to ON to enable coverage tests: these extensive tests don't really need to be done on every single build)
SET (ENABLE_COVERAGE OFF)

# Build type
SET (BUILD_TYPE Release)

# Bring in the settings common to all ctest scripts
include(site_options.cmake)
include(test_setup.cmake)

