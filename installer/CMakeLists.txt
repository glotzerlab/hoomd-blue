# $Id$
# $URL$
# Maintainer: joaander

set(CPACK_DESCRIPTION_SUMMARY "Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition")
set(CPACK_PACKAGE_VENDOR "University of Michigan")
set(CPACK_PACKAGE_VERSION_MAJOR ${HOOMD_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${HOOMD_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${HOOMD_VERSION_PATCH})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "hoomd")

# the package name is upper case now, force it lower here to be pretty
set(CPACK_PACKAGE_NAME "hoomd")

set(CPACK_RESOURCE_FILE_LICENSE "${HOOMD_SOURCE_DIR}/LICENSE")

if (WIN32 AND NOT UNIX)
    set(CPACK_NSIS_INSTALLER_ICON "${HOOMD_SOURCE_DIR}/data/art/InstallIcon.ico")
    set(CPACK_NSIS_URL_INFO_ABOUT "http://codeblue.umich.edu/hoomd-blue")
    set(CPACK_NSIS_CONTACT "http://codeblue.umich.edu/hoomd-blue")
    set(CPACK_NSIS_DISPLAY_NAME "HOOMD-blue")
        
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_PACKAGE_EXECUTABLES "hoomd;HOOMD-blue Python Interpreter")
    
else (WIN32 AND NOT UNIX)
    # on linux, I want to have x86_64 or x86 in the name of the package
    find_program(UNAME_EXE "uname" DOC "uname executable")
    mark_as_advanced(UNAME_EXE)
    if (UNAME_EXE)
        exec_program(${UNAME_EXE} ${HOOMD_SOURCE_DIR} ARGS -m OUTPUT_VARIABLE UNAME_MACHINE)
    else (UNAME_EXE)
        set(UNAME_MACHINE "NA")
    endif (UNAME_EXE)
    
    set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}-${UNAME_MACHINE}")

    # I also want to append the Python version. This method is potentially
    # faulty as it is not necesarily true that the found python interpreter is
    # the same as the one that is linked against. But then, packaging will
    # only be done by me and I know what I'm doing with this.
    find_package(PythonInterp REQUIRED)
    exec_program(${PYTHON_EXECUTABLE} ${HOOMD_SOURCE_DIR} ARGS -V
            OUTPUT_VARIABLE PYTHON_VER_UNFORMATTED)
    # remove the space from the string
    string(REPLACE " " "" PYTHON_VER_NOSPACE ${PYTHON_VER_UNFORMATTED})
    # remove the minor version
    string(SUBSTRING ${PYTHON_VER_NOSPACE} 0 9 PYTHON_VER)

    set(CPACK_SYSTEM_NAME "${CPACK_SYSTEM_NAME}-${PYTHON_VER}")
endif (WIN32 AND NOT UNIX)

if (WIN32)
    set (CPACK_GENERATOR "NSIS")
else (WIN32)
    set (CPACK_GENERATOR "TBZ2")
endif (WIN32)

# use our overrided NSIS template files
# different directories are needed for 64 bit builds vs 32 bit ones
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
if (CMAKE_CL_64)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/x86_64" ${CMAKE_MODULE_PATH})
else (CMAKE_CL_64)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/i386" ${CMAKE_MODULE_PATH})
endif (CMAKE_CL_64)

include(CPack)

# need to install vs dlls on windows
if(WIN32)
include(InstallRequiredSystemLibraries)

# we don't get any feedback if they were installed: check if the redist dir was found
if (MSVC80)
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

find_file(CUDA_CUDART_DLL cudart.dll
        HINTS ${_cudart_search_path})

mark_as_advanced(CUDA_CUDART_DLL)

install(PROGRAMS ${CUDA_CUDART_DLL} DESTINATION bin)
endif (CUDA_FOUND)

endif(WIN32)

