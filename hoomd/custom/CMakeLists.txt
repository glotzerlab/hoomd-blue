set(files __init__.py
          custom_action.py
          custom_operation.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/custom
       )

copy_files_to_build("${files}" "custom" "*.py")
