set(files __init__.py
          potential.py
          bond.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/mesh
       )

copy_files_to_build("${files}" "md-mesh" "*.py")
