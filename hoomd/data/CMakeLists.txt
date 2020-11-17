set(files __init__.py
          array.py
          data_structures.py
          local_access.py
          local_access_cpu.py
          local_access_gpu.py
          param_dict.py
          smart_default.py
          syncedlist.py
          typeconverter.py
          typeparam.py
          type_param_dict.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/data
       )

copy_files_to_build("${files}" "data" "*.py")
