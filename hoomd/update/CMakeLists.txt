set(files __init__.py
          box_resize.py
          remove_drift.py
          custom_updater.py
          particle_filter.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/update
       )

copy_files_to_build("${files}" "update" "*.py")
