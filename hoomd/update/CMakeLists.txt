# set (_header_files)

# install(FILES ${_headers_files}
#         DESTINATION ${PYTHON_SITE_INSTALL_DIR}/include/hoomd/update
#        )

set(files __init__.py
          box_resize.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/update
       )

copy_files_to_build("${files}" "update" "*.py")
