# see ctest_hoomd.cmake for overall documentation
# this version of it tests a configuration different from the default

# (set to ON to enable CUDA build)
SET (ENABLE_CUDA "ON")

# (set to OFF to enable double precision build) (ENABLE_CUDA must be off if this is set off)
SET (SINGLE_PRECISION "ON")

# (set to OFF to enable shared library builds)
SET (ENABLE_STATIC "OFF")

# (set tests to ignore, see the example for the format) 
# (bdnvt and npt take minutes to run, and an enternity with valgrind enabled, so they are ignored by default)
# SET (IGNORE_TESTS "")
SET (IGNORE_TESTS "-E \"bdnvt|npt\"")

# (location of valgrind: Leave blank unless you REALLY want the long valgrind tests to run
SET (MEMORYCHECK_COMMAND "")
#SET (MEMORYCHECK_COMMAND "/usr/bin/valgrind")

# (change to emulation if you want to compile and test a GPU emulation build)
SET (CUDA_BUILD_EMULATION "ON")

# (architecture to compile CUDA for 10=compute 1.0 11=compute 1.1, ...)
SET (CUDA_ARCH "11")

# (set to ON to enable coverage tests: these extensive tests don't really need to be done on every single build)
SET (ENABLE_COVERAGE ON)

# Bring in the settings common to all ctest scripts
include(site_options.cmake)
include(test_setup.cmake)

