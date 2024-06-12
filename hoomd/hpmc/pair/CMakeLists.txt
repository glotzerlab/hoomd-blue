set(files __init__.py
        lennard_jones.py
        expanded_gaussian.py
        pair.py
        step.py
        union.py
        angular_step.py
        user.py
 )

install(FILES ${files}
      DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/pair
     )

copy_files_to_build("${files}" "hpmc-pair" "*.py")
