# List all files that include tests
set(files __init__.py
          test_example_updater.py
    )

# Copy tests to the install directory
install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/updater_plugin/pytest
       )

# Copy tests to the build directory for testing proir to installation
copy_files_to_build("${files}" "updater_plugin_pytest" "*.py")
