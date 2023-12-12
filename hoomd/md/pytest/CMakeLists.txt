# copy python modules to the build directory to make it a working python package
set(files __init__.py
    aniso_forces_and_energies.json
    true_pair_list.json
    test_active.py
    test_active_rotational_diffusion.py
    test_alchemostat.py
    test_alj.py
    test_angle.py
    test_aniso_pair.py
    test_array_view.py
    test_constrain_distance.py
    test_constant_force.py
    test_custom_force.py
    test_external.py
    test_filter_md.py
    test_half_step_hook.py
    test_improper.py
    test_integrate.py
    test_bond.py
    test_dihedral.py
    test_flags.py
    test_kernel_parameters.py
    test_potential.py
    test_pppm_coulomb.py
    test_manifolds.py
    test_meta_wall_list.py
    test_methods.py
    test_meshpotential.py
    test_minimize_fire.py
    test_reverse_perturbation_flow.py
    test_table_pressure.py
    test_thermo.py
    test_thermoHMA.py
    forces_and_energies.json
    test_nlist.py
    test_nlist_tuner.py
    test_rigid.py
    test_zero_momentum.py
    test_gsd.py
    test_special_pair.py
    test_update_group_dof.py
    test_wall_data.py
    test_wall_potential.py
    test_burst_writer.py
    test_hdf5.py
    )

install(FILES ${files}
        DESTINATION ${PYTHON_SITE_INSTALL_DIR}/md/pytest
       )

copy_files_to_build("${files}" "md_pytest" "*.py")
