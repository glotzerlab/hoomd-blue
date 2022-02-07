set(files __init__.py
        user.py
        field.py
 )

install(FILES ${files}
      DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/external
     )

copy_files_to_build("${files}" "hpmc-external" "*.py")
