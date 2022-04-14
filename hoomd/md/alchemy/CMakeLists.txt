set(files __init__.py
          _alchemical_pair_force.py
          pair.py
          methods.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/alchemy
       )

copy_files_to_build("${files}" "md-alchemy" "*.py")
