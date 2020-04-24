set(files __init__.py
          custom_analyzer.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/analyze
       )

copy_files_to_build("${files}" "analyze" "*.py")
