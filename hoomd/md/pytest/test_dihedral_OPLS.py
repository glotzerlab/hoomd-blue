import hoomd
import pytest
import numpy as np

_k1 = [1.0, 0.5, 2.0]
_k2 = [1.5, 2.5, 1.0]
_k3 = [0.5, 1.5, 0.25]
_k4 = [0.75, 1.0, 3.5]


def get_args():
    arg_dicts = []
    for _k1i, _k2i, _k3i, _k4i in zip(_k1, _k2, _k3, _k4):
        arg_dicts.append({'k1': _k1i, 'k2': _k2i, 'k3': _k3i, 'k4': _k4i})
    return arg_dicts


@pytest.mark.parametrize("argument_dict", get_args())
def test_before_attaching(argument_dict):
    dihedral_potential = hoomd.md.dihedral.OPLS()
    dihedral_potential.params['backbone'] = argument_dict
    for key in argument_dict.keys():
        np.testing.assert_allclose(dihedral_potential.params['backbone'][key],
                                   argument_dict[key],
                                   rtol=1e-6)
