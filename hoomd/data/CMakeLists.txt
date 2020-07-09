set(files __init__.py
          local_snapshot.py
          local_snapshot_gpu.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/filter
       )

copy_files_to_build("${files}" "data" "*.py")
