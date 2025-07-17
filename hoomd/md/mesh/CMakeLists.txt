set(files __init__.py
          bending.py
          bond.py
          conservation.py
          potential.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/mesh
       )

copy_files_to_build("${files}" "md-mesh" "*.py")
