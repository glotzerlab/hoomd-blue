# List all files that include python tests
set(files __init__.py
          test_mysphere.py)

# Copy tests to the install directory
install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/shape_plugin/pytest
        )

# Copy tests to the build directory for testing prior to installation
copy_files_to_build("${files}" "shape_plugin_pytest" "*.py")
