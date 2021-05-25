import copy as cp
import numpy as np
import numpy.testing as npt
import pytest

import hoomd


def _evaluate_periodic(snapshot, params):
    box = hoomd.Box(*snapshot.configuration.box)
    positions = snapshot.particles.position
    A = params['A']
    i = params['i']
    w = params['w']
    p = params['p']
    a1, a2, a3 = box.lattice_vectors
    V = np.dot(a1, np.cross(a2, a3))
    b1 = 2 * np.pi / V * np.cross(a2, a3)
    b2 = 2 * np.pi / V * np.cross(a3, a1)
    b3 = 2 * np.pi / V * np.cross(a1, a2)
    b = {0: b1, 1: b2, 2: b3}.get(i)
    energies = A * np.tanh(1/(2*np.pi*p*w) * np.cos(p*np.dot(positions, b)))
    forces = A / (2*np.pi*w) * np.sin(p*np.dot(positions, b))
    forces *= 1 - (np.tanh(np.cos(p*np.dot(positions, b)) / (2 * np.pi * p * w))) **2
    forces = np.outer(forces, b)
    return forces, energies


def _evaluate_electric(snapshot, params):
    box = hoomd.Box(*snapshot.configuration.box)
    positions = snapshot.particles.position
    charges = snapshot.particles.charge
    E_field = params['E']
    energies = -charges * np.dot(positions, E_field)
    forces = np.outer(charges, E_field)
    return forces, energies


def _external_params():
    """ Each entry is a tuple (class_object, list(dictionaries of params, evaluator function)). """
    list_ext_params = []
    list_ext_params.append((hoomd.md.external.Periodic,
                            list([dict(A=1.5, i=1, w=3.5, p=5),
                                 dict(A=10, i=0, w=3.4, p=2)]
                                 ),
                            _evaluate_periodic))
    list_ext_params.append((hoomd.md.external.ElectricField,
                            list([dict(E=(1, 0, 0)),
                                 dict(E=(0, 2, 0)),
                                 ]
                                 ),
                            _evaluate_electric))
    return list_ext_params


@pytest.fixture(scope="function", params=_external_params(), ids=(lambda x: x[0].__name__))
def external_params(request):
    return cp.deepcopy(request.param)


def _assert_correct_params(external_obj, param_dict):
    """ Assert the parameters of the external object match whats in the dictionary. """
    for param in param_dict.keys():
        npt.assert_allclose(external_obj.params['A'][param], param_dict[param])


def test_get_set(simulation_factory, two_particle_snapshot_factory, external_params):
    # unpack parameters
    cls_obj, list_param_dicts, evaluator = external_params

    # create class instance, get/set params when not attached
    obj_instance = cls_obj()
    obj_instance.params['A'] = list_param_dicts[0]
    _assert_correct_params(obj_instance, list_param_dicts[0])
    obj_instance.params['A'] = list_param_dicts[1]
    _assert_correct_params(obj_instance, list_param_dicts[1])

    # set up simulation
    snap = two_particle_snapshot_factory(d=3.7)
    sim = simulation_factory(snap)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.001)
    sim.operations.integrator.forces.append(obj_instance)
    sim.run(0)

    # get/set params while attached
    obj_instance.params['A'] = list_param_dicts[0]
    _assert_correct_params(obj_instance, list_param_dicts[0])
    obj_instance.params['A'] = list_param_dicts[1]
    _assert_correct_params(obj_instance, list_param_dicts[1])

def test_forces_and_energies(simulation_factory, lattice_snapshot_factory, external_params):
    # unpack parameters
    cls_obj, list_param_dicts, evaluator = external_params

    # create class instance, get/set params when not attached
    obj_instance = cls_obj()
    obj_instance.params['A'] = list_param_dicts[0]

    # set up simulation and run a bit
    snap = lattice_snapshot_factory(n=2)
    sim = simulation_factory(snap)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.001)
    sim.operations.integrator.forces.append(obj_instance)
    sim.run(10)

    # test energies
    forces = sim.operations.integrator.forces[0].forces
    energies = sim.operations.integrator.forces[0].energies
    F, E = evaluator(sim.state.snapshot, list_param_dicts[0])
    np.testing.assert_allclose(F, forces)
    np.testing.assert_allclose(E, energies)
