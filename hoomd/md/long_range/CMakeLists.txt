set(files __init__.py
          pppm.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/long_range
       )

copy_files_to_build("${files}" "md-long_range" "*.py")
