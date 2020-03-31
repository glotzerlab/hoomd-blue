set(files __init__.py
          csv_backend.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/output
       )

copy_files_to_build("${files}" "output" "*.py")
