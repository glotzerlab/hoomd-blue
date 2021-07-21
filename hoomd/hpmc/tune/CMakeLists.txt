set(files __init__.py
          move_size.py
          nec_chain_time.py
          )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/tune
       )

copy_files_to_build("${files}" "hpmc_tune" "*.py")
