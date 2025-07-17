set(files __init__.py
        external.py
        harmonic.py
        linear.py
        wall.py
 )

install(FILES ${files}
      DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/external
     )

copy_files_to_build("${files}" "hpmc-external" "*.py")
