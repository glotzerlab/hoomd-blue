# Maintainer: joaander

# Specify cmake modules path to use our custom modules
set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/thrust"
    "${CMAKE_CURRENT_SOURCE_DIR}/hoomd"
    "${CMAKE_CURRENT_SOURCE_DIR}/libgetar"
    "${CMAKE_CURRENT_SOURCE_DIR}/git"
    "${CMAKE_MODULE_PATH}"
    PARENT_SCOPE)
