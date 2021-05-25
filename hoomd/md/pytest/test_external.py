import copy as cp
import numpy.testing as npt
import pytest

import hoomd


def _evaluate_periodic(positions, params):
    pass


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
                                 dict(E=(0, 0, 3)),
                                 ]
                                 ),
                            _evaluate_periodic))
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


