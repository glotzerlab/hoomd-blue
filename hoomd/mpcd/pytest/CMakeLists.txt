# copy python modules to the build directory to make it a working python package
set(files __init__.py
    test_collide.py
    test_fill.py
    test_force.py
    test_geometry.py
    test_integrator.py
    test_methods.py
    test_snapshot.py
    test_stream.py
    test_tune.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/mpcd/pytest
       )

copy_files_to_build("${files}" "mpcd_pytest" "*.py")
