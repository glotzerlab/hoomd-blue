set(files __init__.py
          integrate.py
          tune.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/nec
       )

copy_files_to_build("${files}" "hpmc-nec" "*.py")
