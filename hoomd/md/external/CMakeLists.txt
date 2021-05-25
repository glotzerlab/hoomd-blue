set(files __init__.py
          wall.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/external
       )

copy_files_to_build("${files}" "external" "*.py")

