set(CPACK_NSIS_INSTALLER_ICON "${HOOMD_SOURCE_DIR}/data/art/InstallIcon.ico")
set(CPACK_NSIS_URL_INFO_ABOUT "http://codeblue.umich.edu/hoomd-blue")
set(CPACK_NSIS_CONTACT "http://codeblue.umich.edu/hoomd-blue")
set(CPACK_NSIS_DISPLAY_NAME "HOOMD-blue")

set(CPACK_NSIS_MODIFY_PATH ON)
set(CPACK_PACKAGE_EXECUTABLES "hoomd;HOOMD-blue Python Interpreter")

set (CPACK_GENERATOR "NSIS")

# use our overrided NSIS template files
# different directories are needed for 64 bit builds vs 32 bit ones
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
if (CMAKE_CL_64)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/x86_64" ${CMAKE_MODULE_PATH})
else (CMAKE_CL_64)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/i386" ${CMAKE_MODULE_PATH})
endif (CMAKE_CL_64)

include(CPack)

if (NOT MSVC80_REDIST_DIR)
message("MSVC80 redist dir not found, dlls needed to run HOOMD will not be 
included in the installer. You can safely ignore this message if you do not intend to
build windows installers for HOOMD on this machine.")
endif (NOT MSVC80_REDIST_DIR)
elseif (MSVC90)
if (NOT MSVC90_REDIST_DIR)
message("MSVC90 redist dir not found, dlls needed to run HOOMD will not be 
included in the installer. You can safely ignore this message if you do not intend to
build windows installers for HOOMD on this machine.")
endif (NOT MSVC90_REDIST_DIR)
endif(MSVC80)

# install cudart.dll if CUDA is enabled
if (CUDA_FOUND)
# this finds the dll in bin64 even in 32 bit builds, fix that
if(CMAKE_CL_64)
set (_cudart_search_path ${CUDA_TOOLKIT_ROOT_DIR}/bin64)
else(CMAKE_CL_64)
set (_cudart_search_path ${CUDA_TOOLKIT_ROOT_DIR}/bin)
endif(CMAKE_CL_64)

find_file(CUDA_CUDART_DLL 
        NAMES cudart.dll cudart64_30_14.dll cudart32_30_14.dll
        HINTS ${_cudart_search_path})

mark_as_advanced(CUDA_CUDART_DLL)

install(PROGRAMS ${CUDA_CUDART_DLL} DESTINATION bin)
endif (CUDA_FOUND)
