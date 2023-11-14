set(files __init__.py
          custom_writer.py
          table.py
          gsd.py
          gsd_burst.py
          dcd.py
          hdf5.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/write
       )

copy_files_to_build("${files}" "write" "*.py")
