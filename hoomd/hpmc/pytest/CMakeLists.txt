# copy python modules to the build directory to make it a working python package
MACRO(copy_file file)
    add_custom_command (
        OUTPUT ${file}
        DEPENDS ${file}
        POST_BUILD
        COMMAND    ${CMAKE_COMMAND}
        ARGS       -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file} ${CMAKE_CURRENT_BINARY_DIR}/${file}
        COMMENT    "Copy hoomd/hpmc/pytest/${file}"
    )
ENDMACRO(copy_file)

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
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/pytest
       )

foreach(file ${files})
    copy_file(${file})
endforeach()

add_custom_target(copy_hpmc_pytest ALL DEPENDS ${files})
