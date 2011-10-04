# Maintainer: joaander

if (ENABLE_APP_BUNDLE_INSTALL)
# install the template app directory
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/HOOMD-blue.app DESTINATION "."
        USE_SOURCE_PERMISSIONS
        PATTERN ".svn" EXCLUDE
        )

# configure the info.plist file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Info.plist.in ${CMAKE_CURRENT_BINARY_DIR}/Info.plist)
# and install it
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Info.plist DESTINATION "HOOMD-blue.app/Contents")

if (ENABLE_CUDA)
    get_filename_component(_cuda_libdir ${CUDA_CUDART_LIBRARY} PATH)
endif (ENABLE_CUDA)

set(_libdirs ${Boost_LIBRARY_DIRS} ${_cuda_libdir})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/fixup-bundle.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/fixup-bundle.cmake @ONLY)

install(CODE "include(${CMAKE_CURRENT_BINARY_DIR}/fixup-bundle.cmake)")

# setup CPack for DragNDrop
set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-10.6")
set (CPACK_GENERATOR "DragNDrop")
include(CPack)

endif(ENABLE_APP_BUNDLE_INSTALL)
