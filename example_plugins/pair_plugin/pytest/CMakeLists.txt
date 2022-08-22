# List all files that include python tests.
set(files __init__.py
          test_example_pair.py
    )

# Copy tests to the install directory.
install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/pair_plugin/pytest
       )

# Copy tests to the build directory for testing proir to installation.
copy_files_to_build("${files}" "pair_plugin_pytest" "*.py")
