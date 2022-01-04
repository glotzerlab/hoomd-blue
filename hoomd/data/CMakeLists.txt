set(files __init__.py
          array.py
          collections.py
          local_access.py
          local_access_cpu.py
          local_access_gpu.py
          parameterdicts.py
          smart_default.py
          syncedlist.py
          typeconverter.py
          typeparam.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/data
       )

copy_files_to_build("${files}" "data" "*.py")
