# $Id$
# $URL$
# Maintainer: joaander

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
else (WIN32)
    set(DATA_INSTALL_DIR "share/hoomd")
    set(LIB_INSTALL_DIR "lib/hoomd")
    set(LIB_BASE_INSTALL_DIR "lib")
    set(BIN_INSTALL_DIR "bin")
    set(INC_INSTALL_DIR "include/hoomd")
endif (WIN32)


################################
## Version information
set(HOOMD_VERSION_MAJOR "0")
set(HOOMD_VERSION_MINOR "9")
set(HOOMD_VERSION_PATCH "0")
set(HOOMD_VERSION "${HOOMD_VERSION_MAJOR}.${HOOMD_VERSION_MINOR}.${HOOMD_VERSION_PATCH}")
set(HOOMD_VERSION_LONG "${HOOMD_VERSION}")
if ("$URL$" MATCHES "codeblue.umich.edu/hoomd-blue/svn/tags")
set(HOOMD_SUBVERSION_BUILD "0")
else ("$URL$" MATCHES "codeblue.umich.edu/hoomd-blue/svn/tags")
set(HOOMD_SUBVERSION_BUILD "1")
endif ("$URL$" MATCHES "codeblue.umich.edu/hoomd-blue/svn/tags")

# users may not have subversion installed, search for it and set an unknown version if that is
# the case
find_program(SVNVERSION_EXE NAMES "git-svnversion.sh" "svnversion" DOC "svnversion executable")
mark_as_advanced(SVNVERSION_EXE)
if (SVNVERSION_EXE)
    exec_program(${SVNVERSION_EXE} ${HOOMD_SOURCE_DIR} ARGS ${HOOMD_SOURCE_DIR} OUTPUT_VARIABLE SVNVERSION)
else (SVNVERSION_EXE)
    set(SVNVERSION "NA")
endif (SVNVERSION_EXE)

if(HOOMD_SUBVERSION_BUILD)
    string(REGEX REPLACE ":" "_" SVNVERSION_UNDERSCORE ${SVNVERSION})
    set(CPACK_PACKAGE_VERSION "${HOOMD_VERSION}_beta${SVNVERSION_UNDERSCORE}")
    set(HOOMD_VERSION_LONG "${CPACK_PACKAGE_VERSION}")
endif(HOOMD_SUBVERSION_BUILD)



