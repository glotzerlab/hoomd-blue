set(files __init__.py
          molecules.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/molecules
       )

copy_files_to_build("${files}" "md-molecules" "*.py")
