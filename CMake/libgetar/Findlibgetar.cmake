find_path(libgetar_DIR
  NAMES src/GTAR.hpp src/SharedArray.hpp
  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/hoomd/extern/libgetar
  DOC "Location of the libgetar root directory"
  NO_DEFAULT_PATH)

find_path(LIBGETAR_SRC_DIR
  NAMES GTAR.hpp SharedArray.hpp
  HINTS ${libgetar_DIR}/src
  DOC "Location of libgetar's src/ directory"
  NO_DEFAULT_PATH)

mark_as_advanced(LIBGETAR_SRC_DIR)
mark_as_advanced(libgetar_DIR)
