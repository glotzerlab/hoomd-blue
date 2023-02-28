# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import (pickling_check, logging_check,
                            autotuned_kernel_parameter_check)
from hoomd.logging import LoggerCategories
import pytest
from copy import deepcopy
from collections import namedtuple

paramtuple = namedtuple('paramtuple', [
    'setup_params', 'extra_params', 'changed_params', 'rattle_method', 'method'
])


def test_validate_group():
    CUBE_VERTS = [
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
    ]

    rigid = hoomd.md.constrain.Rigid()
    rigid.body['R'] = {
        "constituent_types": ['A'] * 8,
        "positions": CUBE_VERTS,
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 8,
    }

    nve1 = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    nve2 = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0,
                                     methods=[nve1, nve2],
                                     integrate_rotational_dof=True)
    integrator.rigid = rigid

    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_gsd('init.gsd')

    sim.operations.integrator = integrator

    with pytest.raises(RuntimeError):
        sim.run(10)


def _method_base_params():
    method_base_params_list = []
    # Start with valid parameters to get the keys and placeholder values

    langevin_setup_params = {'kT': hoomd.variant.Constant(2.0)}
    langevin_extra_params = {'alpha': None, 'tally_reservoir_energy': False}
    langevin_changed_params = {
        'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000),
        'alpha': None,
        'tally_reservoir_energy': True
    }

    method_base_params_list.extend([
        paramtuple(langevin_setup_params, langevin_extra_params,
                   langevin_changed_params, hoomd.md.methods.rattle.Langevin,
                   hoomd.md.methods.Langevin)
    ])

    brownian_setup_params = {'kT': hoomd.variant.Constant(2.0)}
    brownian_extra_params = {'alpha': None}
    brownian_changed_params = {
        'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000),
        'alpha': 0.125
    }

    method_base_params_list.extend([
        paramtuple(brownian_setup_params, brownian_extra_params,
                   brownian_changed_params, hoomd.md.methods.rattle.Brownian,
                   hoomd.md.methods.Brownian)
    ])

    overdamped_viscous_setup_params = {}
    overdamped_viscous_extra_params = {'alpha': None}
    overdamped_viscous_changed_params = {'alpha': 0.125}

    method_base_params_list.extend([
        paramtuple(overdamped_viscous_setup_params,
                   overdamped_viscous_extra_params,
                   overdamped_viscous_changed_params,
                   hoomd.md.methods.rattle.OverdampedViscous,
                   hoomd.md.methods.OverdampedViscous)
    ])

    constant_s = [
        hoomd.variant.Constant(1.0),
        hoomd.variant.Constant(2.0),
        hoomd.variant.Constant(3.0),
        hoomd.variant.Constant(0.125),
        hoomd.variant.Constant(.25),
        hoomd.variant.Constant(.5)
    ]

    ramp_s = [
        hoomd.variant.Ramp(1.0, 4.0, 1000, 10000),
        hoomd.variant.Ramp(2.0, 4.0, 1000, 10000),
        hoomd.variant.Ramp(3.0, 4.0, 1000, 10000),
        hoomd.variant.Ramp(0.125, 4.0, 1000, 10000),
        hoomd.variant.Ramp(.25, 4.0, 1000, 10000),
        hoomd.variant.Ramp(.5, 4.0, 1000, 10000)
    ]

    npt_setup_params = {
        'kT': hoomd.variant.Constant(2.0),
        'tau': 2.0,
        'S': constant_s,
        'tauS': 2.0,
        'box_dof': [True, True, True, False, False, False],
        'couple': 'xyz'
    }
    npt_extra_params = {
        'rescale_all': False,
        'gamma': 0.0,
        'translational_thermostat_dof': (0.0, 0.0),
        'rotational_thermostat_dof': (0.0, 0.0),
        'barostat_dof': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
    npt_changed_params = {
        'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000),
        'tau': 10.0,
        'S': ramp_s,
        'tauS': 10.0,
        'box_dof': [True, False, False, False, True, False],
        'couple': 'none',
        'rescale_all': True,
        'gamma': 2.0,
        'translational_thermostat_dof': (0.125, 0.5),
        'rotational_thermostat_dof': (0.5, 0.25),
        'barostat_dof': (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    }

    method_base_params_list.extend([
        paramtuple(npt_setup_params, npt_extra_params, npt_changed_params, None,
                   hoomd.md.methods.NPT)
    ])

    nvt_setup_params = {'kT': hoomd.variant.Constant(2.0), 'tau': 2.0}
    nvt_extra_params = {}
    nvt_changed_params = {
        'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000),
        'tau': 10.0,
        'translational_thermostat_dof': (0.125, 0.5),
        'rotational_thermostat_dof': (0.5, 0.25)
    }

    method_base_params_list.extend([
        paramtuple(nvt_setup_params, nvt_extra_params, nvt_changed_params, None,
                   hoomd.md.methods.NVT)
    ])

    nve_setup_params = {}
    nve_extra_params = {}
    nve_changed_params = {}

    method_base_params_list.extend([
        paramtuple(nve_setup_params, nve_extra_params, nve_changed_params,
                   hoomd.md.methods.rattle.NVE, hoomd.md.methods.NVE)
    ])

    displacement_capped_setup_params = {
        "maximum_displacement": hoomd.variant.Ramp(1e-3, 1e-1, 0, 1_00)
    }

    displacement_capped_extra_params = {}
    displacement_capped_changed_params = {
        "maximum_displacement": hoomd.variant.Constant(1e-2)
    }

    method_base_params_list.extend([
        paramtuple(displacement_capped_setup_params,
                   displacement_capped_extra_params,
                   displacement_capped_changed_params,
                   hoomd.md.methods.rattle.DisplacementCapped,
                   hoomd.md.methods.DisplacementCapped)
    ])

    return method_base_params_list


@pytest.fixture(scope="function",
                params=_method_base_params(),
                ids=(lambda x: x[4].__name__))
def method_base_params(request):
    return deepcopy(request.param)


def check_instance_attrs(instance, attr_dict, set_attrs=False):
    for attr, value in attr_dict.items():
        if set_attrs:
            setattr(instance, attr, value)
        if hasattr(value, "__iter__") and not isinstance(value, str):
            assert all(v == instance_v
                       for v, instance_v in zip(value, getattr(instance, attr)))
        else:
            assert getattr(instance, attr) == value


def test_attributes(method_base_params):
    all_ = hoomd.filter.All()
    method = method_base_params.method(**method_base_params.setup_params,
                                       filter=all_)

    assert method.filter is all_

    check_instance_attrs(method, method_base_params.setup_params)
    check_instance_attrs(method, method_base_params.extra_params)

    type_A = hoomd.filter.Type(['A'])
    method.filter = type_A
    assert method.filter is type_A

    check_instance_attrs(method, method_base_params.changed_params, True)


def test_attributes_attached(simulation_factory, two_particle_snapshot_factory,
                             method_base_params):

    all_ = hoomd.filter.All()
    method = method_base_params.method(**method_base_params.setup_params,
                                       filter=all_)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
    sim.run(0)

    assert method.filter is all_

    check_instance_attrs(method, method_base_params.setup_params)
    check_instance_attrs(method, method_base_params.extra_params)

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        method.filter = type_A

    check_instance_attrs(method, method_base_params.changed_params, True)


def test_switch_methods(simulation_factory, two_particle_snapshot_factory,
                        method_base_params):

    all_ = hoomd.filter.All()
    method = method_base_params.method(**method_base_params.setup_params,
                                       filter=all_)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
    sim.run(5)

    sim.operations.integrator.methods.remove(method)

    assert len(sim.operations.integrator.methods) == 0

    sim.operations.integrator.methods.append(hoomd.md.methods.NVE(all_))

    assert len(sim.operations.integrator.methods) == 1

    sim.run(5)


def _manifold_base_params():
    manifold_base_params_list = []
    # Start with valid parameters to get the keys and placeholder values

    cylinder_setup_params = {'r': 5}
    manifold_base_params_list.extend([
        paramtuple(cylinder_setup_params, {}, {}, {},
                   hoomd.md.manifold.Cylinder)
    ])

    diamond_setup_params = {'N': (1, 1, 1)}
    manifold_base_params_list.extend([
        paramtuple(diamond_setup_params, {}, {}, {}, hoomd.md.manifold.Diamond)
    ])

    ellipsoid_setup_params = {'a': 3.3, 'b': 5, 'c': 4.1}

    manifold_base_params_list.extend([
        paramtuple(ellipsoid_setup_params, {}, {}, {},
                   hoomd.md.manifold.Ellipsoid)
    ])

    gyroid_setup_params = {'N': (1, 2, 1)}

    manifold_base_params_list.extend(
        [paramtuple(gyroid_setup_params, {}, {}, {}, hoomd.md.manifold.Gyroid)])

    primitive_setup_params = {'N': (1, 1, 1)}

    manifold_base_params_list.extend([
        paramtuple(primitive_setup_params, {}, {}, {},
                   hoomd.md.manifold.Primitive)
    ])

    sphere_setup_params = {'r': 5}

    manifold_base_params_list.extend(
        [paramtuple(sphere_setup_params, {}, {}, {}, hoomd.md.manifold.Sphere)])

    xyplane_setup_params = {}

    manifold_base_params_list.extend(
        [paramtuple(xyplane_setup_params, {}, {}, {}, hoomd.md.manifold.Plane)])

    return manifold_base_params_list


@pytest.fixture(scope="function",
                params=_manifold_base_params(),
                ids=(lambda x: x[4].__name__))
def manifold_base_params(request):
    return deepcopy(request.param)


def test_rattle_attributes(method_base_params, manifold_base_params):
    if method_base_params.rattle_method is None:
        pytest.skip("RATTLE method is not implemented for this method")

    all_ = hoomd.filter.All()
    manifold = manifold_base_params.method(**manifold_base_params.setup_params)
    method = method_base_params.rattle_method(**method_base_params.setup_params,
                                              filter=all_,
                                              manifold_constraint=manifold)
    assert method.manifold_constraint == manifold
    assert method.tolerance == 1e-6

    sphere = hoomd.md.manifold.Sphere(r=10)
    with pytest.raises(AttributeError):
        method.manifold_constraint = sphere
    assert method.manifold_constraint == manifold

    method.tolerance = 1e-5
    assert method.tolerance == 1e-5


def test_rattle_attributes_attached(simulation_factory,
                                    two_particle_snapshot_factory,
                                    method_base_params, manifold_base_params):

    if method_base_params.rattle_method is None:
        pytest.skip("RATTLE integrator is not implemented for this method")

    all_ = hoomd.filter.All()
    manifold = manifold_base_params.method(**manifold_base_params.setup_params)
    method = method_base_params.rattle_method(**method_base_params.setup_params,
                                              filter=all_,
                                              manifold_constraint=manifold)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
    sim.run(0)

    assert method.filter is all_
    assert method.manifold_constraint == manifold
    assert method.tolerance == 1e-6

    check_instance_attrs(method, method_base_params.setup_params)
    check_instance_attrs(method, method_base_params.extra_params)

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        method.filter = type_A

    sphere = hoomd.md.manifold.Sphere(r=10)
    with pytest.raises(AttributeError):
        # manifold cannot be set after scheduling
        method.manifold_constraint = sphere
    assert method.manifold_constraint == manifold

    method.tolerance = 1e-5
    assert method.tolerance == 1e-5

    check_instance_attrs(method, method_base_params.changed_params, True)


def test_rattle_switch_methods(simulation_factory,
                               two_particle_snapshot_factory,
                               method_base_params, manifold_base_params):

    if method_base_params.rattle_method is None:
        pytest.skip("RATTLE integrator is not implemented for this method")

    all_ = hoomd.filter.All()
    manifold = manifold_base_params.method(**manifold_base_params.setup_params)
    method = method_base_params.rattle_method(**method_base_params.setup_params,
                                              filter=all_,
                                              manifold_constraint=manifold)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
    sim.run(5)

    sim.operations.integrator.methods.remove(method)

    assert len(sim.operations.integrator.methods) == 0

    sim.operations.integrator.methods.append(
        hoomd.md.methods.rattle.NVE(filter=all_, manifold_constraint=manifold))

    assert len(sim.operations.integrator.methods) == 1

    sim.run(5)


def test_nph_attributes_attached_3d(simulation_factory,
                                    two_particle_snapshot_factory):
    """Test attributes of the NPH integrator after attaching in 3D."""
    all_ = hoomd.filter.All()
    constant_s = [
        hoomd.variant.Constant(1.0),
        hoomd.variant.Constant(2.0),
        hoomd.variant.Constant(3.0),
        hoomd.variant.Constant(0.125),
        hoomd.variant.Constant(.25),
        hoomd.variant.Constant(.5)
    ]
    nph = hoomd.md.methods.NPH(filter=all_,
                               S=constant_s,
                               tauS=2.0,
                               couple='xyz')

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nph])
    sim.run(0)

    assert nph.filter == all_
    assert len(nph.S) == 6
    for i in range(6):
        assert nph.S[i] is constant_s[i]
    assert nph.tauS == 2.0
    assert nph.couple == 'xyz'

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        nph.filter = type_A

    assert nph.filter == all_

    nph.tauS = 10.0
    assert nph.tauS == 10.0

    box_dof = (True, False, False, False, True, False)
    nph.box_dof = box_dof
    assert nph.box_dof == box_dof

    nph.couple = 'none'
    assert nph.couple == 'none'

    nph.rescale_all = True
    assert nph.rescale_all

    nph.gamma = 2.0
    assert nph.gamma == 2.0

    assert nph.barostat_dof == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    nph.barostat_dof = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    assert nph.barostat_dof == (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)

    ramp_s = [
        hoomd.variant.Ramp(1.0, 4.0, 1000, 10000),
        hoomd.variant.Ramp(2.0, 4.0, 1000, 10000),
        hoomd.variant.Ramp(3.0, 4.0, 1000, 10000),
        hoomd.variant.Ramp(0.125, 4.0, 1000, 10000),
        hoomd.variant.Ramp(.25, 4.0, 1000, 10000),
        hoomd.variant.Ramp(.5, 4.0, 1000, 10000)
    ]
    nph.S = ramp_s
    assert len(nph.S) == 6
    for _ in range(5):
        sim.run(1)
        for i in range(6):
            assert nph.S[i] is ramp_s[i]


def test_npt_thermalize_thermostat_and_barostat_dof(
        simulation_factory, two_particle_snapshot_factory):
    """Tests that NPT.thermalize_thermostat_and_barostat_dof can be called."""
    all_ = hoomd.filter.All()
    constant_t = hoomd.variant.Constant(2.0)
    constant_s = [1, 2, 3, 0.125, 0.25, 0.5]
    npt = hoomd.md.methods.NPT(filter=all_,
                               kT=constant_t,
                               tau=2.0,
                               S=constant_s,
                               tauS=2.0,
                               box_dof=[True, True, True, True, True, True],
                               couple='xyz')

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[npt])
    sim.run(0)

    npt.thermalize_thermostat_and_barostat_dof()
    xi, eta = npt.translational_thermostat_dof
    assert xi != 0.0
    assert eta == 0.0

    xi_rot, eta_rot = npt.rotational_thermostat_dof
    assert xi_rot == 0.0
    assert eta_rot == 0.0

    for v in npt.barostat_dof:
        assert v != 0.0


def test_npt_thermalize_thermostat_and_barostat_aniso_dof(
        simulation_factory, two_particle_snapshot_factory):
    """Tests that NPT.thermalize_thermostat_and_barostat_dof can be called."""
    all_ = hoomd.filter.All()
    constant_t = hoomd.variant.Constant(2.0)
    constant_s = [1, 2, 3, 0.125, 0.25, 0.5]
    npt = hoomd.md.methods.NPT(filter=all_,
                               kT=constant_t,
                               tau=2.0,
                               S=constant_s,
                               tauS=2.0,
                               box_dof=[True, True, True, True, True, True],
                               couple='xyz')

    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.moment_inertia[:] = [[1, 1, 1], [2, 0, 0]]

    sim = simulation_factory(snap)

    sim.operations.integrator = hoomd.md.Integrator(
        0.005, methods=[npt], integrate_rotational_dof=True)
    sim.run(0)

    npt.thermalize_thermostat_and_barostat_dof()
    xi, eta = npt.translational_thermostat_dof
    assert xi != 0.0
    assert eta == 0.0

    xi_rot, eta_rot = npt.rotational_thermostat_dof
    assert xi_rot != 0.0
    assert eta_rot == 0.0
    for v in npt.barostat_dof:
        assert v != 0.0


def test_nph_thermalize_barostat_dof(simulation_factory,
                                     two_particle_snapshot_factory):
    """Tests that NPT.thermalize_thermostat_and_barostat_dof can be called."""
    all_ = hoomd.filter.All()
    constant_s = [1, 2, 3, 0.125, 0.25, 0.5]
    nph = hoomd.md.methods.NPH(filter=all_,
                               S=constant_s,
                               tauS=2.0,
                               box_dof=[True, True, True, True, True, True],
                               couple='xyz')

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nph])
    sim.run(0)

    nph.thermalize_barostat_dof()
    for v in nph.barostat_dof:
        assert v != 0.0


def test_npt_attributes_attached_2d(simulation_factory,
                                    two_particle_snapshot_factory):
    """Test attributes of the NPT integrator specific to 2D simulations."""
    all_ = hoomd.filter.All()
    npt = hoomd.md.methods.NPT(filter=all_,
                               kT=1.0,
                               tau=2.0,
                               S=2.0,
                               tauS=2.0,
                               couple='xy')

    assert npt.box_dof == [True, True, True, False, False, False]
    assert npt.couple == 'xy'

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=2))
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[npt])
    sim.run(0)

    # after attaching in 2d, only some coupling modes and box dof are valid
    assert npt.box_dof == [True, True, False, False, False, False]
    assert npt.couple == 'xy'

    with pytest.raises(ValueError):
        npt.couple = 'xyz'
    with pytest.raises(ValueError):
        npt.couple = 'xz'
    with pytest.raises(ValueError):
        npt.couple = 'yz'

    npt.couple = 'none'
    assert npt.couple == 'none'

    npt.box_dof = [True, True, True, True, True, True]
    assert npt.box_dof == [True, True, False, True, False, False]


def test_nvt_thermalize_thermostat_dof(simulation_factory,
                                       two_particle_snapshot_factory):
    """Tests that NVT.thermalize_thermostat_dof can be called."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    nvt = hoomd.md.methods.NVT(filter=all_, kT=constant, tau=2.0)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nvt])
    sim.run(0)

    nvt.thermalize_thermostat_dof()
    xi, eta = nvt.translational_thermostat_dof
    assert xi != 0.0
    assert eta == 0.0

    xi_rot, eta_rot = nvt.rotational_thermostat_dof
    assert xi_rot == 0.0
    assert eta_rot == 0.0


def test_nvt_thermalize_thermostat_aniso_dof(simulation_factory,
                                             two_particle_snapshot_factory):
    """Tests that NVT.thermalize_thermostat_dof can be called."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    nvt = hoomd.md.methods.NVT(filter=all_, kT=constant, tau=2.0)

    snap = two_particle_snapshot_factory()
    if snap.communicator.rank == 0:
        snap.particles.moment_inertia[:] = [[1, 1, 1], [2, 0, 0]]

    sim = simulation_factory(snap)
    sim.operations.integrator = hoomd.md.Integrator(
        0.005, methods=[nvt], integrate_rotational_dof=True)
    sim.run(0)

    nvt.thermalize_thermostat_dof()
    xi, eta = nvt.translational_thermostat_dof
    assert xi != 0.0
    assert eta == 0.0

    xi_rot, eta_rot = nvt.rotational_thermostat_dof
    assert xi_rot != 0.0
    assert eta_rot == 0.0


def test_kernel_parameters(method_base_params, simulation_factory,
                           two_particle_snapshot_factory):
    method = method_base_params.method(**method_base_params.setup_params,
                                       filter=hoomd.filter.All())

    sim = simulation_factory(two_particle_snapshot_factory())
    if (method_base_params.method == hoomd.md.methods.Berendsen
            and sim.device.communicator.num_ranks > 1):
        pytest.skip("Berendsen method does not support multiple processor "
                    "configurations.")
    integrator = hoomd.md.Integrator(0.05, methods=[method])
    sim.operations.integrator = integrator
    sim.run(0)

    autotuned_kernel_parameter_check(instance=method,
                                     activate=lambda: sim.run(1))


def test_pickling(method_base_params, simulation_factory,
                  two_particle_snapshot_factory):
    method = method_base_params.method(**method_base_params.setup_params,
                                       filter=hoomd.filter.All())

    pickling_check(method)
    sim = simulation_factory(two_particle_snapshot_factory())
    if (method_base_params.method == hoomd.md.methods.Berendsen
            and sim.device.communicator.num_ranks > 1):
        pytest.skip("Berendsen method does not support multiple processor "
                    "configurations.")
    integrator = hoomd.md.Integrator(0.05, methods=[method])
    sim.operations.integrator = integrator
    sim.run(0)
    pickling_check(method)


def test_logging():
    logging_check(hoomd.md.methods.NPH, ('md', 'methods'), {
        'barostat_energy': {
            'category': LoggerCategories.scalar,
            'default': True
        }
    })
    logging_check(
        hoomd.md.methods.NPT, ('md', 'methods'), {
            'barostat_energy': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'thermostat_energy': {
                'category': LoggerCategories.scalar,
                'default': True
            }
        })
    logging_check(hoomd.md.methods.NVT, ('md', 'methods'), {
        'thermostat_energy': {
            'category': LoggerCategories.scalar,
            'default': True
        }
    })


@pytest.mark.parametrize("cls, init_args", [
    (hoomd.md.methods.Brownian, {
        'kT': 1.5
    }),
    (hoomd.md.methods.Langevin, {
        'kT': 1.5
    }),
    (hoomd.md.methods.OverdampedViscous, {}),
])
def test_default_gamma(cls, init_args):
    c = cls(filter=hoomd.filter.All(), **init_args)
    assert c.gamma['A'] == 1.0
    assert c.gamma_r['A'] == (1.0, 1.0, 1.0)

    c = cls(filter=hoomd.filter.All(), **init_args, default_gamma=2.0)
    assert c.gamma['A'] == 2.0
    assert c.gamma_r['A'] == (1.0, 1.0, 1.0)

    c = cls(filter=hoomd.filter.All(),
            **init_args,
            default_gamma_r=(3.0, 4.0, 5.0))
    assert c.gamma['A'] == 1.0
    assert c.gamma_r['A'] == (3.0, 4.0, 5.0)


def test_langevin_reservoir(simulation_factory, two_particle_snapshot_factory):

    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.5)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    methods=[langevin])
    sim.run(10)
    assert langevin.reservoir_energy == 0.0

    langevin.tally_reservoir_energy = True

    sim.run(10)
    assert langevin.reservoir_energy != 0.0
