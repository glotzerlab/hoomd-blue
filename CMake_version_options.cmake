# Maintainer: joaander

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
