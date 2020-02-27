# copy python modules to the build directory to make it a working python package
set(files __init__.py
    test_sphere.py
	test_ellipsoid.py
    test_convex_polygon.py
    test_convex_polyhedron.py
    test_polyhedron.py
    test_sphinx.py
    test_faceted_ellipsoid.py
    test_simple_polygon.py
    test_convex_spheropolygon.py
    test_convex_spheropolyhedron.py
    test_sphere_union.py
    test_spheropolyhedron_union.py
    test_faceted_ellipsoid_union.py
    test_shape.py
    conftest.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/hpmc/pytest
       )

copy_files_to_build("${files}" "hpmc_pytest" "*.py")
