import hoomd
import pytest
import numpy
import itertools
from copy import deepcopy
from collections import namedtuple

paramtuple = namedtuple('paramtuple',
                        ['setup_params',
                         'extra_params',
                         'integrator_params',
                         'integrator'])


def _integrator_base_params():
    integrator_base_params_list = []
    # Start with valid parameters to get the keys and placeholder values

    langevin_dict1 = {'kT': hoomd.variant.Constant(2.0), 'seed': 2 }
    langevin_dict2 = {'alpha': None, 'tally_reservoir_energy': False }
    langevin_dict3 = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000), 
                      'seed': 23, 'alpha': None, 'tally_reservoir_energy': True }

    integrator_base_params_list.extend([paramtuple(langevin_dict1,
                                                    langevin_dict2,
                                                    langevin_dict3,
                                                    hoomd.md.methods.Langevin)])

    brownian_dict1 = {'kT': hoomd.variant.Constant(2.0), 'seed': 2 }
    brownian_dict2 = {'alpha': None }
    brownian_dict3 = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000), 
                      'seed': 108, 'alpha': 0.125}

    integrator_base_params_list.extend([paramtuple(brownian_dict1,
                                                    brownian_dict2,
                                                    brownian_dict3,
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

    npt_dict1 = {'kT': hoomd.variant.Constant(2.0), 'tau': 2.0, 'S': constant_s, 
            'tauS': 2.0, 'box_dof': (True,True,True,False,False,False), 'couple': 'xyz' }
    npt_dict2 = {'rescale_all': False, 'gamma': 0.0, 'translational_thermostat_dof': (0.0,0.0), 
            'rotational_thermostat_dof': (0.0, 0.0), 'barostat_dof': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) }
    npt_dict3 = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000), 'tau': 10.0, 'S': ramp_s, 
            'tauS': 10.0, 'box_dof': (True,False,False,False,True,False), 'couple': 'none', 
            'rescale_all': True, 'gamma': 2.0, 'translational_thermostat_dof': (0.125, 0.5), 
            'rotational_thermostat_dof': (0.5, 0.25), 'barostat_dof': (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)}

    integrator_base_params_list.extend([paramtuple(npt_dict1,
                                                    npt_dict2,
                                                    npt_dict3,
                                                    hoomd.md.methods.NPT)])

    nvt_dict1 = {'kT': hoomd.variant.Constant(2.0), 'tau': 2.0 }
    nvt_dict2 = { }
    nvt_dict3 = {'kT': hoomd.variant.Ramp(1, 2, 1000000, 2000000), 'tau': 10.0,
            'translational_thermostat_dof': (0.125, 0.5), 'rotational_thermostat_dof': (0.5, 0.25)}

    integrator_base_params_list.extend([paramtuple(nvt_dict1,
                                                    nvt_dict2,
                                                    nvt_dict3,
                                                    hoomd.md.methods.NVT)])

    nve_dict1 = { }
    nve_dict2 = { }
    nve_dict3 = { }

    integrator_base_params_list.extend([paramtuple(nve_dict1,
                                                    nve_dict2,
                                                    nve_dict3,
                                                    hoomd.md.methods.NVE)])

    return integrator_base_params_list


@pytest.fixture(scope="function", params=_integrator_base_params(), ids=(lambda x: x[3].__name__))
def integrator_base_params(request):
    return deepcopy(request.param)

def test_attributes(integrator_base_params):

    all_ = hoomd.filter.All()
    integrator = integrator_base_params.integrator(**integrator_base_params.setup_params,filter=all_)
    
    assert integrator.filter is all_

    for pair in integrator_base_params.setup_params:
        attr = integrator_base_params.setup_params[pair]
        if isinstance(attr,list):
            list_attr = integrator._getattr_param(pair)
            assert len(list_attr) == 6
            for i in range(6):
                assert list_attr[i] == attr[i]
        else:
            assert integrator._getattr_param(pair) == attr

    for pair in integrator_base_params.extra_params:
        attr = integrator_base_params.extra_params[pair]
        assert integrator._getattr_param(pair) == attr

    type_A = hoomd.filter.Type(['A'])
    integrator.filter = type_A
    assert integrator.filter is type_A

    for pair in integrator_base_params.integrator_params:
        attr = integrator_base_params.integrator_params[pair]
        integrator._setattr_param(pair,attr)
        if isinstance(attr,list):
            list_attr = integrator._getattr_param(pair)
            assert len(list_attr) == 6
            for i in range(6):
                assert list_attr[i] == attr[i]
        else:
            assert integrator._getattr_param(pair) == attr


def test_brownian_attributes_attached(simulation_factory,
                                      two_particle_snapshot_factory):
    """Test attributes of the Brownian integrator after attaching."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    brownian = hoomd.md.methods.Brownian(filter = all_, kT=constant, seed=2)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[brownian])
    sim.run(0)

    assert brownian.filter is all_
    assert brownian.kT is constant
    assert brownian.seed == 2
    assert brownian.alpha is None
    assert brownian.manifold_constraint is None

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        brownian.filter = type_A

    assert brownian.filter is all_

    ramp = hoomd.variant.Ramp(1, 2, 1000000, 2000000)
    brownian.kT = ramp
    assert brownian.kT is ramp

    with pytest.raises(AttributeError):
        # seed cannot be set after scheduling
        brownian.seed = 10

    assert brownian.seed == 2

    brownian.alpha = 0.125
    assert brownian.alpha == 0.125

@pytest.mark.parametrize("integrator", [hoomd.md.methods.Brownian, hoomd.md.methods.Langevin])
def test_rattle_attributes(integrator):
    """Test attributes of the LangevinBase RATTLE integrator before attaching."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    gyroid = hoomd.md.manifold.Gyroid(N=1)
    rattle = integrator(filter = all_, kT=constant, manifold_constraint = gyroid, seed=2, eta = 1e-5)

    assert rattle.filter is all_
    assert rattle.kT is constant
    assert rattle.manifold_constraint is gyroid
    assert rattle.seed == 2
    assert rattle.alpha is None
    assert rattle.eta == 1e-5

    type_A = hoomd.filter.Type(['A'])
    rattle.filter = type_A
    assert rattle.filter is type_A

    ramp = hoomd.variant.Ramp(1, 2, 1000000, 2000000)
    rattle.kT = ramp
    assert rattle.kT is ramp

    sphere = hoomd.md.manifold.Sphere(r=10)
    rattle.manifold_constraint = sphere
    assert rattle.manifold_constraint is sphere

    rattle.seed = 10
    assert rattle.seed == 10

    rattle.eta = 0.001
    assert rattle.eta == 0.001

    rattle.alpha = 0.125
    assert rattle.alpha == 0.125


@pytest.mark.parametrize("integrator", [hoomd.md.methods.Brownian, hoomd.md.methods.Langevin])
def test_rattle_attributes_attached(simulation_factory,
                                      two_particle_snapshot_factory,
                                      integrator):
    """Test attributes of the LanfgevinBase RATTLE integrator after attaching."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    plane = hoomd.md.manifold.Plane()
    rattle = integrator(filter = all_, kT=constant, manifold_constraint = plane, seed=2)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[rattle])
    sim.run(0)

    assert rattle.filter is all_
    assert rattle.kT is constant
    assert rattle.manifold_constraint is plane
    assert rattle.seed == 2
    assert rattle.alpha is None
    assert rattle.eta == 1e-6

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        rattle.filter = type_A

    assert rattle.filter is all_

    ramp = hoomd.variant.Ramp(1, 2, 1000000, 2000000)
    rattle.kT = ramp
    assert rattle.kT is ramp

    gyroid = hoomd.md.manifold.Gyroid(N=2)
    with pytest.raises(AttributeError):
        # manifold cannot be set after scheduling
        rattle.manifold_constraint = gyroid

    assert rattle.manifold_constraint is plane

    with pytest.raises(AttributeError):
        # seed cannot be set after scheduling
        rattle.seed = 10

    assert rattle.seed == 2

    rattle.eta = 0.001
    assert rattle.eta == 0.001

    rattle.alpha = 0.125
    assert rattle.alpha == 0.125



def test_langevin_attributes_attached(simulation_factory,
                                      two_particle_snapshot_factory):
    """Test attributes of the Langevin integrator before attaching."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    langevin = hoomd.md.methods.Langevin(filter = all_, kT=constant, seed=2)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[langevin])
    sim.run(0)

    assert langevin.filter is all_
    assert langevin.kT is constant
    assert langevin.seed == 2
    assert langevin.alpha is None
    assert (not langevin.tally_reservoir_energy)
    assert langevin.manifold_constraint is None

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        langevin.filter = type_A

    assert langevin.filter is all_

    ramp = hoomd.variant.Ramp(1, 2, 1000000, 2000000)
    langevin.kT = ramp
    assert langevin.kT is ramp

    with pytest.raises(AttributeError):
        # seed cannot be set after scheduling
        langevin.seed = 10

    assert langevin.seed == 2

    langevin.alpha = 0.125
    assert langevin.alpha == 0.125

    langevin.tally_reservoir_energy = True
    assert langevin.tally_reservoir_energy

def test_langevin_rattle_attributes():
    """Test attributes of the Langevin RATTLE integrator before attaching."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    gyroid = hoomd.md.manifold.Gyroid(N=1)
    langevin_rattle = hoomd.md.methods.Langevin(filter = all_, kT=constant, manifold_constraint = gyroid, seed=2, eta=1e-5)

    assert langevin_rattle.filter is all_
    assert langevin_rattle.kT is constant
    assert langevin_rattle.manifold_constraint is gyroid
    assert langevin_rattle.seed == 2
    assert langevin_rattle.eta == 1e-5
    assert langevin_rattle.alpha is None
    assert (not langevin_rattle.tally_reservoir_energy)

    type_A = hoomd.filter.Type(['A'])
    langevin_rattle.filter = type_A
    assert langevin_rattle.filter is type_A

    ramp = hoomd.variant.Ramp(1, 2, 1000000, 2000000)
    langevin_rattle.kT = ramp
    assert langevin_rattle.kT is ramp

    sphere = hoomd.md.manifold.Sphere(r=10)
    langevin_rattle.manifold_constraint = sphere
    assert langevin_rattle.manifold_constraint == sphere

    langevin_rattle.seed = 10
    assert langevin_rattle.seed == 10

    langevin_rattle.eta = 0.001
    assert langevin_rattle.eta == 0.001

    langevin_rattle.alpha = 0.125
    assert langevin_rattle.alpha == 0.125

    langevin_rattle.tally_reservoir_energy = True
    assert langevin_rattle.tally_reservoir_energy


def test_langevin_rattle_attributes_attached(simulation_factory,
                                      two_particle_snapshot_factory):
    """Test attributes of the Langevin RATTLE integrator before attaching."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    plane = hoomd.md.manifold.Plane()
    langevin_rattle = hoomd.md.methods.Langevin(filter = all_, kT=constant, manifold_constraint = plane, seed=2)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[langevin_rattle])
    sim.run(0)

    assert langevin_rattle.filter is all_
    assert langevin_rattle.kT is constant
    assert langevin_rattle.manifold_constraint == plane
    assert langevin_rattle.seed == 2
    assert langevin_rattle.eta == 1e-6
    assert langevin_rattle.alpha is None
    assert (not langevin_rattle.tally_reservoir_energy)

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        langevin_rattle.filter = type_A

    assert langevin_rattle.filter is all_

    ramp = hoomd.variant.Ramp(1, 2, 1000000, 2000000)
    langevin_rattle.kT = ramp
    assert langevin_rattle.kT is ramp

    gyroid = hoomd.md.manifold.Gyroid(N=2)
    with pytest.raises(AttributeError):
        # manifold cannot be set after scheduling
        langevin_rattle.manifold_constraint = gyroid

    with pytest.raises(AttributeError):
        # seed cannot be set after scheduling
        langevin_rattle.seed = 10

    assert langevin_rattle.seed == 2

    langevin_rattle.alpha = 0.125
    assert langevin_rattle.alpha == 0.125

    langevin_rattle.eta = 0.001
    assert langevin_rattle.eta == 0.001

    langevin_rattle.tally_reservoir_energy = True
    assert langevin_rattle.tally_reservoir_energy


def test_npt_attributes_attached_3d(simulation_factory,
                                      two_particle_snapshot_factory):
    """Test attributes of the NPT integrator before attaching."""
    all_ = hoomd.filter.All()
    constant_t = hoomd.variant.Constant(2.0)
    constant_s = [hoomd.variant.Constant(1.0),
                  hoomd.variant.Constant(2.0),
                  hoomd.variant.Constant(3.0),
                  hoomd.variant.Constant(0.125),
                  hoomd.variant.Constant(.25),
                  hoomd.variant.Constant(.5)]
    npt = hoomd.md.methods.NPT(filter = all_, kT=constant_t, tau=2.0,
                               S = constant_s,
                               tauS = 2.0,
                               couple='xyz')

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[npt])
    sim.run(0)

    assert npt.filter is all_
    assert npt.kT is constant_t
    assert npt.tau == 2.0
    assert len(npt.S) == 6
    for i in range(6):
        assert npt.S[i] is constant_s[i]
    assert npt.tauS == 2.0
    assert npt.couple == 'xyz'

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        npt.filter = type_A

    assert npt.filter is all_

    ramp = hoomd.variant.Ramp(1, 2, 1000000, 2000000)
    npt.kT = ramp
    assert npt.kT is ramp

    npt.tau = 10.0
    assert npt.tau == 10.0

    ramp_s = [hoomd.variant.Ramp(1.0, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(2.0, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(3.0, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(0.125, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(.25, 4.0, 1000, 10000),
                  hoomd.variant.Ramp(.5, 4.0, 1000, 10000)]
    npt.S = ramp_s
    assert len(npt.S) == 6
    for i in range(6):
        assert npt.S[i] is ramp_s[i]

    npt.tauS = 10.0
    assert npt.tauS == 10.0

    npt.box_dof = (True,False,False,False,True,False)
    assert tuple(npt.box_dof) == (True,False,False,False,True,False)

    npt.couple = 'none'
    assert npt.couple == 'none'

    npt.rescale_all = True
    assert npt.rescale_all

    npt.gamma = 2.0
    assert npt.gamma == 2.0

    assert npt.translational_thermostat_dof == (0.0, 0.0)
    npt.translational_thermostat_dof = (0.125, 0.5)
    assert npt.translational_thermostat_dof == (0.125, 0.5)

    assert npt.rotational_thermostat_dof == (0.0, 0.0)
    npt.rotational_thermostat_dof = (0.5, 0.25)
    assert npt.rotational_thermostat_dof == (0.5, 0.25)

    assert npt.barostat_dof == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    npt.barostat_dof = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    assert npt.barostat_dof == (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)


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

    npt.thermalize_thermostat_and_barostat_dof(100)
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

    npt.thermalize_thermostat_and_barostat_dof(100)
    xi, eta = npt.translational_thermostat_dof
    assert xi != 0.0
    assert eta == 0.0

    xi_rot, eta_rot = npt.rotational_thermostat_dof
    assert xi_rot != 0.0
    assert eta_rot == 0.0
    for v in npt.barostat_dof:
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


def test_nve_attributes_attached(simulation_factory,
                                 two_particle_snapshot_factory):
    """Test attributes of the NVE integrator after attaching."""
    all_ = hoomd.filter.All()
    nve = hoomd.md.methods.NVE(filter = all_)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nve])
    sim.run(0)

    assert nve.filter is all_

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        nve.filter = type_A

    assert nve.filter is all_

def test_nve_rattle_attributes():
    """Test attributes of the NVE RATTLE integrator before attaching."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    gyroid = hoomd.md.manifold.Gyroid(N=1)
    nve_rattle = hoomd.md.methods.NVE(filter = all_, manifold_constraint = gyroid, eta = 1e-5)

    assert nve_rattle.filter is all_
    assert nve_rattle.manifold_constraint is gyroid
    assert nve_rattle.eta == 1e-5

    type_A = hoomd.filter.Type(['A'])
    nve_rattle.filter = type_A
    assert nve_rattle.filter is type_A

    sphere = hoomd.md.manifold.Sphere(r=10)
    nve_rattle.manifold_constraint = sphere
    assert nve_rattle.manifold_constraint is sphere

    nve_rattle.eta = 0.001
    assert nve_rattle.eta == 0.001


def test_nve_rattle_attributes_attached(simulation_factory,
                                 two_particle_snapshot_factory):
    """Test attributes of the NVE RATTLE integrator after attaching."""
    all_ = hoomd.filter.All()
    plane = hoomd.md.manifold.Plane()
    nve_rattle = hoomd.md.methods.NVE(filter = all_, manifold_constraint = plane)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nve_rattle])
    sim.run(0)

    assert nve_rattle.filter is all_
    assert nve_rattle.manifold_constraint is plane
    assert nve_rattle.eta == 1e-6

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        nve_rattle.filter = type_A

    assert nve_rattle.filter is all_

    gyroid = hoomd.md.manifold.Gyroid(N=2)
    with pytest.raises(AttributeError):
        # manifold cannot be set after scheduling
        nve_rattle.manifold_constraint = gyroid

    nve_rattle.eta = 0.001
    assert nve_rattle.eta == 0.001

def test_nvt_attributes_attached(simulation_factory,
                                      two_particle_snapshot_factory):
    """Test attributes of the NVT integrator before attaching."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    nvt = hoomd.md.methods.NVT(filter = all_, kT=constant, tau=2.0)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nvt])
    sim.run(0)

    assert nvt.filter is all_
    assert nvt.kT is constant
    assert nvt.tau == 2.0

    type_A = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        nvt.filter = type_A

    assert nvt.filter is all_

    ramp = hoomd.variant.Ramp(1, 2, 1000000, 2000000)
    nvt.kT = ramp
    assert nvt.kT is ramp

    nvt.tau = 10.0
    assert nvt.tau == 10.0

    assert nvt.translational_thermostat_dof == (0.0, 0.0)
    nvt.translational_thermostat_dof = (0.125, 0.5)
    assert nvt.translational_thermostat_dof == (0.125, 0.5)

    assert nvt.rotational_thermostat_dof == (0.0, 0.0)
    nvt.rotational_thermostat_dof = (0.5, 0.25)
    assert nvt.rotational_thermostat_dof == (0.5, 0.25)


def test_nvt_thermalize_thermostat_dof(simulation_factory,
                                       two_particle_snapshot_factory):
    """Tests that NVT.thermalize_thermostat_dof can be called."""
    all_ = hoomd.filter.All()
    constant = hoomd.variant.Constant(2.0)
    nvt = hoomd.md.methods.NVT(filter=all_, kT=constant, tau=2.0)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nvt])
    sim.run(0)

    nvt.thermalize_thermostat_dof(100)
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

    nvt.thermalize_thermostat_dof(100)
    xi, eta = nvt.translational_thermostat_dof
    assert xi != 0.0
    assert eta == 0.0

    xi_rot, eta_rot = nvt.rotational_thermostat_dof
    assert xi_rot != 0.0
    assert eta_rot == 0.0
