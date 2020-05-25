set(files __init__.py
          custom_updater.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/update
       )

copy_files_to_build("${files}" "update" "*.py")
