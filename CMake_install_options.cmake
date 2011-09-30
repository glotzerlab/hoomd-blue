# $Id$
# $URL$
# Maintainer: joaander

# option to build Mac OS X .app packages
if (APPLE)
option(ENABLE_APP_BUNDLE_INSTALL "Enable installation of an app bundle installation" OFF)
endif(APPLE)

option(ENABLE_EMBED_CUDA "Enable embedding of the CUDA libraries into lib/hoomd" OFF)
mark_as_advanced(ENABLE_EMBED_CUDA)

# setup flags to specify installation directories for files, these differ in
# linux and windows
if (WIN32)
    # The "." needs to be there to install to the root directory of
    # the specified install path, "" doesn't work
    set(DATA_INSTALL_DIR ".")
    set(LIB_INSTALL_DIR "bin")
    set(LIB_BASE_INSTALL_DIR "bin")
    set(BIN_INSTALL_DIR "bin")
    set(INC_INSTALL_DIR "include/hoomd")
elseif (ENABLE_APP_BUNDLE_INSTALL)
    set(DATA_INSTALL_DIR "HOOMD-blue.app/Contents/share/hoomd")
    set(LIB_INSTALL_DIR "HOOMD-blue.app/Contents/lib/hoomd")
    set(LIB_BASE_INSTALL_DIR "HOOMD-blue.app/Contents/lib")
    set(BIN_INSTALL_DIR "HOOMD-blue.app/Contents/MacOS")
    set(INC_INSTALL_DIR "HOOMD-blue.app/Contents/include/hoomd")
else (WIN32)
    set(DATA_INSTALL_DIR "share/hoomd")
    set(LIB_INSTALL_DIR "lib${LIB_SUFFIX}/hoomd")
    set(LIB_BASE_INSTALL_DIR "lib${LIB_SUFFIX}")
    set(BIN_INSTALL_DIR "bin")
    set(INC_INSTALL_DIR "include/hoomd")
endif (WIN32)


################################
## Version information
set(HOOMD_VERSION_MAJOR "0")
set(HOOMD_VERSION_MINOR "9")
set(HOOMD_VERSION_PATCH "2")
set(HOOMD_VERSION "${HOOMD_VERSION_MAJOR}.${HOOMD_VERSION_MINOR}.${HOOMD_VERSION_PATCH}")

# users may not have git installed, or this may be a tarball build - set a dummy version if that is the case
include(GetGitRevisionDescription)
git_describe(HOOMD_GIT_VERSION)
if (HOOMD_GIT_VERSION)
    set(HOOMD_VERSION_LONG "${HOOMD_GIT_VERSION}")
else (HOOMD_GIT_VERSION)
    set(HOOMD_VERSION_LONG "${HOOMD_VERSION}-unknown")
endif (HOOMD_GIT_VERSION)
