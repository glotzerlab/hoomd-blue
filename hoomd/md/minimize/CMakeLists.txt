################ Python only modules
# copy python modules to the build directory to make it a working python package
set(files __init__.py
          fire.py
   )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/minimize
       )

copy_files_to_build("${files}" "minimize" "*.py")
