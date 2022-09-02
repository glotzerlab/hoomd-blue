set(files __init__.py
          bond.py
          bending.py
          conservation.py
          potential.py
          surface_tension.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/mesh
       )

copy_files_to_build("${files}" "md-mesh" "*.py")
