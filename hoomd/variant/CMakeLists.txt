set(files __init__.py
          box.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/variant
       )

copy_files_to_build("${files}" "variant" "*.py")
