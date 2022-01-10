set(files __init__.py
          field.py
          wall.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/external
       )

copy_files_to_build("${files}" "md-external" "*.py")
