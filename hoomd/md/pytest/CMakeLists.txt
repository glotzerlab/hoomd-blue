# copy python modules to the build directory to make it a working python package
set(files __init__.py
    test_flags.py
    test_pair_potential.py
    test_pair.py
    test_methods.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/pytest
       )

copy_files_to_build("${files}" "md_pytest" "*.py")
