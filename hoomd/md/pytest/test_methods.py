import hoomd
import pytest
import numpy
import itertools
from copy import deepcopy
from collections import namedtuple

paramtuple = namedtuple('paramtuple',
                        ['setup_params',
                         'extra_params',
                         'accepted_params',
                         'rejected_params',
                         'method'])


def _method_base_params():
    method_base_params_list = []
    # Start with valid parameters to get the keys and placeholder values

    langevin_setup_params = {'kT': hoomd.variant.Constant(2.0) }
    langevin_extra_params = {'alpha': None, 'tally_reservoir_energy': False }
    langevin_accepted_params = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000),
                      'alpha': None, 'tally_reservoir_energy': True }
    langevin_rejected_params = {}

    method_base_params_list.extend([paramtuple(langevin_setup_params,
                                                    langevin_extra_params,
                                                    langevin_accepted_params,
                                                    langevin_rejected_params,
                                                    hoomd.md.methods.Langevin)])

    brownian_setup_params = {'kT': hoomd.variant.Constant(2.0) }
    brownian_extra_params = {'alpha': None }
    brownian_accepted_params = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000),
                      'alpha': 0.125}
    brownian_rejected_params = {}

    method_base_params_list.extend([paramtuple(brownian_setup_params,
                                                    brownian_extra_params,
                                                    brownian_accepted_params,
                                                    brownian_rejected_params,
                                                    hoomd.md.methods.Brownian)])

    constant_s = [hoomd.variant.Constant(1.0),
                  hoomd.variant.Constant(2.0),
                  hoomd.variant.Constant(3.0),
                  hoomd.variant.Constant(0.125),
                  hoomd.variant.Constant(.25),
                  hoomd.variant.Constant(.5)]

    ramp_s = [hoomd.variant.Ramp(1.0, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(2.0, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(3.0, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(0.125, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(.25, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(.5, 4.0, 1000, 10000)]

    npt_setup_params = {'kT': hoomd.variant.Constant(2.0), 'tau': 2.0, 'S': constant_s,
            'tauS': 2.0, 'box_dof': (True,True,True,False,False,False), 'couple': 'xyz' }
    npt_extra_params = {'rescale_all': False, 'gamma': 0.0, 'translational_thermostat_dof': (0.0,0.0),
            'rotational_thermostat_dof': (0.0, 0.0), 'barostat_dof': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) }
    npt_accepted_params = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000), 'tau': 10.0, 'S': ramp_s,
            'tauS': 10.0, 'box_dof': (True,False,False,False,True,False), 'couple': 'none',
            'rescale_all': True, 'gamma': 2.0, 'translational_thermostat_dof': (0.125, 0.5),
            'rotational_thermostat_dof': (0.5, 0.25), 'barostat_dof': (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)}
    npt_rejected_params = { }

    method_base_params_list.extend([paramtuple(npt_setup_params,
                                                    npt_extra_params,
                                                    npt_accepted_params,
                                                    npt_rejected_params,
                                                    hoomd.md.methods.NPT)])

    nvt_setup_params = {'kT': hoomd.variant.Constant(2.0), 'tau': 2.0 }
    nvt_extra_params = { }
    nvt_accepted_params = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000), 'tau': 10.0,
            'translational_thermostat_dof': (0.125, 0.5), 'rotational_thermostat_dof': (0.5, 0.25)}
    nvt_rejected_params = { }

    method_base_params_list.extend([paramtuple(nvt_setup_params,
                                                    nvt_extra_params,
                                                    nvt_accepted_params,
                                                    nvt_rejected_params,
                                                    hoomd.md.methods.NVT)])

    nve_setup_params = { }
    nve_extra_params = { }
    nve_accepted_params = { }
    nve_rejected_params = { }

    method_base_params_list.extend([paramtuple(nve_setup_params,
                                                    nve_extra_params,
                                                    nve_accepted_params,
                                                    nve_rejected_params,
                                                    hoomd.md.methods.NVE)])

    return method_base_params_list


@pytest.fixture(scope="function", params=_method_base_params(), ids=(lambda x: x[4].__name__))
def method_base_params(request):
    return deepcopy(request.param)

def test_attributes(method_base_params):

    all_ = hoomd.filter.All()
    integrator = method_base_params.method(**method_base_params.setup_params,filter=all_)

    assert integrator.filter is all_

    for pair in method_base_params.setup_params:
        attr = method_base_params.setup_params[pair]
        if isinstance(attr,list):
            list_attr = integrator._getattr_param(pair)
            assert len(list_attr) == 6
            for i in range(6):
                assert list_attr[i] == attr[i]
        else:
            assert integrator._getattr_param(pair) == attr

    for pair in method_base_params.extra_params:
        attr = method_base_params.extra_params[pair]
        assert integrator._getattr_param(pair) == attr

    type_A = hoomd.filter.Type(['A'])
    integrator.filter = type_A
    assert integrator.filter is type_A

    for pair in method_base_params.rejected_params:
        attr = method_base_params.rejected_params[pair]
        integrator._setattr_param(pair,attr)
        assert integrator._getattr_param(pair) == attr

    for pair in method_base_params.accepted_params:
        attr = method_base_params.accepted_params[pair]
        integrator._setattr_param(pair,attr)
        if isinstance(attr,list):
            list_attr = integrator._getattr_param(pair)
            assert len(list_attr) == 6
            for i in range(6):
                assert list_attr[i] == attr[i]
        else:
            assert integrator._getattr_param(pair) == attr

def test_attributes_attached(simulation_factory,
                                two_particle_snapshot_factory,
                                method_base_params):

    all_ = hoomd.filter.All()
    integrator = method_base_params.method(**method_base_params.setup_params,filter=all_)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[integrator])
    sim.run(0)

    assert integrator.filter is all_

    for pair in method_base_params.setup_params:
        attr = method_base_params.setup_params[pair]
        if isinstance(attr,list) or isinstance(attr, tuple):
            list_attr = integrator._getattr_param(pair)
            assert len(list_attr) == 6
            for i in range(6):
                assert list_attr[i] == attr[i]
        else:
            assert integrator._getattr_param(pair) == attr

    for pair in method_base_params.extra_params:
        attr = method_base_params.extra_params[pair]
        assert integrator._getattr_param(pair) == attr



    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        integrator.filter = type_A

    for pair in method_base_params.rejected_params:
        attr = method_base_params.rejected_params[pair]
        with pytest.raises(AttributeError):
            # cannot be set after scheduling
            integrator._setattr_param(pair,attr)
        assert integrator._getattr_param(pair) == integrator_base_params.setup_params[pair]


    for pair in method_base_params.accepted_params:
        attr = method_base_params.accepted_params[pair]
        integrator._setattr_param(pair,attr)
        if isinstance(attr, list) or isinstance(attr, tuple):
            list_attr = integrator._getattr_param(pair)
            for i in range(len(attr)):
                assert list_attr[i] == attr[i]
        else:
            assert integrator._getattr_param(pair) == attr



def _rattle_base_params():
    method_base_params_list = []
    # Start with valid parameters to get the keys and placeholder values

    langevin_setup_params = {'kT': hoomd.variant.Constant(2.0) }
    langevin_extra_params = {'alpha': None, 'eta': 1e-6, 'tally_reservoir_energy': False }
    langevin_accepted_params = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000),
            'alpha': None, 'eta': 1e-5, 'tally_reservoir_energy': True }
    langevin_rejected_params = {}

    method_base_params_list.extend([paramtuple(langevin_setup_params,
                                                    langevin_extra_params,
                                                    langevin_accepted_params,
                                                    langevin_rejected_params,
                                                    hoomd.md.methods.Langevin)])

    brownian_setup_params = {'kT': hoomd.variant.Constant(2.0) }
    brownian_extra_params = {'alpha': None,  'eta': 1e-6}
    brownian_accepted_params = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000),
            'alpha': 0.125, 'eta': 1e-5}
    brownian_rejected_params = {}

    method_base_params_list.extend([paramtuple(brownian_setup_params,
                                                    brownian_extra_params,
                                                    brownian_accepted_params,
                                                    brownian_rejected_params,
                                                    hoomd.md.methods.Brownian)])

    nve_setup_params = { }
    nve_extra_params = { 'eta': 1e-6}
    nve_accepted_params = { 'eta': 1e-5}
    nve_rejected_params = { }

    method_base_params_list.extend([paramtuple(nve_setup_params,
                                                    nve_extra_params,
                                                    nve_accepted_params,
                                                    nve_rejected_params,
                                                    hoomd.md.methods.NVE)])

    return method_base_params_list


@pytest.fixture(scope="function", params=_rattle_base_params(), ids=(lambda x: x[4].__name__))
def rattle_base_params(request):
    return deepcopy(request.param)

def test_rattle_attributes(rattle_base_params):

    all_ = hoomd.filter.All()
    gyroid = hoomd.md.manifold.Gyroid(N=1)
    integrator = rattle_base_params.method(**rattle_base_params.setup_params,filter=all_, manifold_constraint = gyroid)

    assert integrator.filter is all_
    assert integrator.manifold_constraint is gyroid

    for pair in rattle_base_params.setup_params:
        attr = rattle_base_params.setup_params[pair]
        assert integrator._getattr_param(pair) == attr

    for pair in rattle_base_params.extra_params:
        attr = rattle_base_params.extra_params[pair]
        assert integrator._getattr_param(pair) == attr

    type_A = hoomd.filter.Type(['A'])
    integrator.filter = type_A
    assert integrator.filter is type_A

    sphere = hoomd.md.manifold.Sphere(r=10)
    integrator.manifold_constraint = sphere
    assert integrator.manifold_constraint is sphere

    for pair in rattle_base_params.rejected_params:
        attr = rattle_base_params.rejected_params[pair]
        integrator._setattr_param(pair,attr)
        assert integrator._getattr_param(pair) == attr

    for pair in rattle_base_params.accepted_params:
        attr = rattle_base_params.accepted_params[pair]
        integrator._setattr_param(pair,attr)
        assert integrator._getattr_param(pair) == attr

def test_rattle_attributes_attached(simulation_factory,
                                two_particle_snapshot_factory,
                                rattle_base_params):

    all_ = hoomd.filter.All()
    gyroid = hoomd.md.manifold.Gyroid(N=1)
    integrator = rattle_base_params.method(**rattle_base_params.setup_params,filter=all_, manifold_constraint = gyroid)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[integrator])
    sim.run(0)

    assert integrator.filter is all_
    assert integrator.manifold_constraint is gyroid

    for pair in rattle_base_params.setup_params:
        attr = rattle_base_params.setup_params[pair]
        assert integrator._getattr_param(pair) == attr

    for pair in rattle_base_params.extra_params:
        attr = rattle_base_params.extra_params[pair]
        assert integrator._getattr_param(pair) == attr

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        integrator.filter = type_A

    sphere = hoomd.md.manifold.Sphere(r=10)
    with pytest.raises(AttributeError):
        # manifold cannot be set after scheduling
        integrator.manifold_constraint = sphere
    assert integrator.manifold_constraint is gyroid

    for pair in rattle_base_params.rejected_params:
        attr = rattle_base_params.rejected_params[pair]
        with pytest.raises(AttributeError):
            # cannot be set after scheduling
            integrator._setattr_param(pair,attr)
        assert integrator._getattr_param(pair) == rattle_base_params.setup_params[pair]


    for pair in rattle_base_params.accepted_params:
        attr = rattle_base_params.accepted_params[pair]
        integrator._setattr_param(pair,attr)
        assert integrator._getattr_param(pair) == attr



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

    nph.box_dof = (True, False, False, False, True, False)
    assert tuple(nph.box_dof) == (True, False, False, False, True, False)

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
    if snap.exists:
        snap.particles.moment_inertia[:] = [[1, 1, 1], [2, 0, 0]]

    sim = simulation_factory(snap)

    sim.operations.integrator = hoomd.md.Integrator(0.005,
                                                    methods=[npt],
                                                    aniso=True)
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
    npt = hoomd.md.methods.NPT(filter = all_, kT=1.0, tau=2.0,
                               S = 2.0,
                               tauS = 2.0,
                               couple='xy')


    assert npt.box_dof == (True,True,True,False,False,False)
    assert npt.couple == 'xy'

    sim = simulation_factory(two_particle_snapshot_factory(dimensions=2))
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[npt])
    sim.run(0)

    # after attaching in 2d, only some coupling modes and box dof are valid
    assert tuple(npt.box_dof) == (True,True,False,False,False,False)
    assert npt.couple == 'xy'

    with pytest.raises(ValueError):
        npt.couple = 'xyz'
    with pytest.raises(ValueError):
        npt.couple = 'xz'
    with pytest.raises(ValueError):
        npt.couple = 'yz'

    npt.couple = 'none'
    assert npt.couple == 'none'

    npt.box_dof = (True, True, True, True, True, True)
    assert tuple(npt.box_dof) == (True, True, False, True, False, False)


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
    if snap.exists:
        snap.particles.moment_inertia[:] = [[1, 1, 1], [2, 0, 0]]

    sim = simulation_factory(snap)
    sim.operations.integrator = hoomd.md.Integrator(0.005,
                                                    methods=[nvt],
                                                    aniso=True)
    sim.run(0)

    nvt.thermalize_thermostat_dof()
    xi, eta = nvt.translational_thermostat_dof
    assert xi != 0.0
    assert eta == 0.0

    xi_rot, eta_rot = nvt.rotational_thermostat_dof
    assert xi_rot != 0.0
    assert eta_rot == 0.0
