# copy python modules to the build directory to make it a working python package
set(files __init__.py
    test_collide.py
    test_integrator.py
    test_snapshot.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd/pytest
       )

copy_files_to_build("${files}" "mpcd_pytest" "*.py")
