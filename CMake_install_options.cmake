# Maintainer: joaander

option(ENABLE_EMBED_CUDA "Enable embedding of the CUDA libraries into lib/hoomd" OFF)
mark_as_advanced(ENABLE_EMBED_CUDA)

set(LIB_INSTALL_DIR "lib${LIB_SUFFIX}/hoomd")
set(LIB_BASE_INSTALL_DIR "lib${LIB_SUFFIX}")
set(BIN_INSTALL_DIR "bin")
set(INC_INSTALL_DIR "include/hoomd")
