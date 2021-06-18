# copy python modules to the build directory to make it a working python package
set(files __init__.py
    aniso_forces_and_energies.json
    test_active.py
    test_angle.py
    test_aniso_pair.py
    test_bond.py
    test_dihedral.py
    test_flags.py
    test_lj_equation_of_state.py
    test_potential.py
    test_manifolds.py
    test_methods.py
    test_reverse_perturbation_flow.py
    test_thermo.py
    test_thermoHMA.py
    forces_and_energies.json
    test_write_debug_data_md.py
    test_nlist.py
    test_rigid.py
    test_zero_momentum.py
    test_gsd.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/pytest
       )

copy_files_to_build("${files}" "md_pytest" "*.py")
