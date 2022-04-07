set(files __init__.py
          pair.py
          aniso.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/pair
       )

copy_files_to_build("${files}" "md-pair" "*.py")
