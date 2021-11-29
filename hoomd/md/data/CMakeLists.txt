set(files __init__.py
          local_access.py
          local_access_cpu.py
          local_access_gpu.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/data
       )

copy_files_to_build("${files}" "md-data" "*.py")
