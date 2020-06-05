# copy python modules to the build directory to make it a working python package
set(files __init__.py
          test_box.py
          test_device.py
          test_example.py
          test_trigger.py
          test_type_parameter_dict.py
          test_typeparam.py
          test_operation.py
          test_syncedlist.py
          test_triggeredops.py
          test_local_snapshot.py
          test_logger.py
          test_filter.py
          dummy.py
          test_snapshot.py
          test_state.py
          test_simulation.py
          test_variant.py
          pytest-openmpi.sh
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/pytest
       )

copy_files_to_build("${files}" "hoomd_pytest" "*.py")
