set(files __init__.py
          methods.py
          rattle.py
          thermostats.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/methods
       )

copy_files_to_build("${files}" "md-methods" "*.py")
