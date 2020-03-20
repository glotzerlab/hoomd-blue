################ Python only modules
# copy python modules to the build directory to make it a working python package
set(files __init__.py
          filter_.py
          all_.py
          set_.py
          tags.py
          type_.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/filter
       )

copy_files_to_build("${files}" "filter" "*.py")
