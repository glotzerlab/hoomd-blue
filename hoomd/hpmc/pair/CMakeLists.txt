set(files __init__.py
        lennard_jones.py
        expanded_gaussian.py
        lj_gauss.py
	    opp.py
	    pair.py
        step.py
        union.py
        angular_step.py
 )

install(FILES ${files}
      DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/pair
     )

copy_files_to_build("${files}" "hpmc-pair" "*.py")
