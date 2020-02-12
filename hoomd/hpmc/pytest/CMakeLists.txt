# copy python modules to the build directory to make it a working python package
set(files __init__.py
    	test_sphere.py
	test_ellipsoid.py
    test_convex_polygon.py
    test_convex_polyhedron.py
    test_polyhedron.py
    test_sphinx.py
    test_faceted_ellipsoid.py
    test_union.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/pytest
       )

copy_files_to_build("${files}" "hpmc_pytest" "*.py")
