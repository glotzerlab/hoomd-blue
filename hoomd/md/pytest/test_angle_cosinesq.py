import hoomd
import pytest
import numpy as np

_k = [3.0, 10.0, 5.0]
_t0 = [np.pi / 2, np.pi / 4, np.pi / 6]


def get_args():
    return [{'k': _ki, 't0': _t0i} for _ki, _t0i in zip(_k, _t0)]


@pytest.mark.parametrize("argument_dict", get_args())
def test_before_attaching(argument_dict):
    angle_potential = hoomd.md.angle.Cosinesq()
    angle_potential.params['backbone'] = argument_dict
    for key in argument_dict.keys():
        np.testing.assert_allclose(angle_potential.params['backbone'][key],
                                   argument_dict[key],
                                   rtol=1e-6)
