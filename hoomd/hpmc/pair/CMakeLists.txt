set(files __init__.py
        user.py
 )

install(FILES ${files}
      DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/pair
     )

copy_files_to_build("${files}" "hpmc-pair" "*.py")
