set(files __init__.py
          table.py
          gsd.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/write
       )

copy_files_to_build("${files}" "write" "*.py")
