# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/pytest/${file}"
    )
ENDMACRO(copy_file)

set(files __init__.py
          test_device.py
          test_example.py
          test_trigger.py
          test_type_parameter_dict.py
          test_typeparam.py
          test_operation.py
          test_syncedlist.py
          test_triggeredops.py
          test_logger.py
          dummy.py
          test_snapshot.py
          test_state.py
          test_variant.py
          pytest-openmpi.sh
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/pytest
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_hoomd_pytest ALL DEPENDS ${files})
