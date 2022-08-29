# copy python modules to the build directory to make it a working python package
set(files __init__.py
          nlist_buffer.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/tune
       )

copy_files_to_build("${files}" "md-tune" "*.py")
