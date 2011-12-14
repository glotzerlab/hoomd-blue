## This file should be placed in the root directory of your project.
## Then modify the CMakeLists.txt file in the root directory of your
## project to incorporate the testing dashboard.
## # The following are required to uses Dart and the Cdash dashboard
##   ENABLE_TESTING()
##   INCLUDE(CTest)
set(CTEST_PROJECT_NAME "HOOMD")
set(CTEST_NIGHTLY_START_TIME "23:00:00 CST")

set(CTEST_DROP_METHOD "http")

# this convoluted setup seems to be the only way to get the SITE_NAME variable
set(SITE ${SITE_NAME})

# the automated test on nyx need to be ssh-tunneled out of the compute node
if (SITE MATCHES "nyx")
set(CTEST_DROP_SITE "localhost:8080")
else (SITE MATCHES "nyx")
set(CTEST_DROP_SITE "cdash.fourpisolutions.com")
endif(SITE MATCHES "nyx")

set(CTEST_DROP_LOCATION "/submit.php?project=HOOMD")
set(CTEST_DROP_SITE_CDASH TRUE)
