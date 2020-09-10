# Maintainer: jglaser
# Find Thrust header files
# based on a source from ryuuta@gmail.com

find_path( THRUST_INCLUDE_DIR
    HINTS /usr/include/cuda /usr/local/include ${CUDA_INCLUDE_DIRS}
    NAMES thrust/version.h
    DOC "Thrust headers"
)

# Find thrust version
file( STRINGS ${THRUST_INCLUDE_DIR}/thrust/version.h
      _version_line
      REGEX "#define THRUST_VERSION[ \t]+([0-9]+)"
)

string(REGEX REPLACE "#define THRUST_VERSION[ \t]+([0-9]+).*$" "\\1" _version "${_version_line}")
math( EXPR major ${_version}/100000 )
math( EXPR minor ${_version}/100%1000 )
math( EXPR subminor ${_version}%100 )
set( THRUST_VERSION "${major}.${minor}.${subminor}" )

# Check for required components
set( THRUST_FOUND TRUE )

find_package_handle_standard_args( Thrust REQUIRED_VARS THRUST_INCLUDE_DIR VERSION_VAR THRUST_VERSION )
