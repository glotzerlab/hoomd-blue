## ctest_site_options.cmake contains common options for all ctest scripts of varying configurations
## it is home to all of the various options that are set once for each site processing the tests
## See ctest_hoomd.cmake for complete documentation on setting up and running tests

## Modify these variables to meet the specific requirements of your site

# (location where hoomd src and bin directory are)
SET (CTEST_CHECKOUT_DIR "$ENV{HOME}/hoomd-dart-tests/hoomd")
SET (CTEST_SOURCE_DIRECTORY "${CTEST_CHECKOUT_DIR}")
SET (CTEST_BINARY_DIRECTORY "${CTEST_CHECKOUT_DIR}/../build_ctest")

# (Experimental is for testing the test script itself, Nightly is for production tests)
SET (TEST_GROUP "Experimental")
# SET (TEST_GROUP "Nightly")

# (name of computer performing the tests)
SET (SITE_NAME "sitename")

# (name of hoomd branch you are testing)
SET (HOOMD_BRANCH "master")

# (name of the system)
set (SYSTEM_NAME "Gentoo")

# (a string identifying the compiler: this cannot be autodetected here)
SET (COMPILER_NAME "gcc434")

###################
## Usually, you don't need to modify the below, but they are site specific options
# other stuff that you might want to modify
SET (CTEST_GIT_COMMAND "git")
SET (CTEST_CMAKE_COMMAND "cmake")
SET (CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE)
