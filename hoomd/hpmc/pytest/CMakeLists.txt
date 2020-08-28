# copy python modules to the build directory to make it a working python package
set(files __init__.py
          test_shape.py
          test_move_size_tuner.py
          test_quick_compress.py
          conftest.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/pytest
       )

copy_files_to_build("${files}" "hpmc_pytest" "*.py")
