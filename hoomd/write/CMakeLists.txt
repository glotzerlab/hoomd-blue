set(files __init__.py
          custom_writer.py
          table.py
          gsd.py
          gtar.py
          dcd.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/write
       )

copy_files_to_build("${files}" "write" "*.py")
