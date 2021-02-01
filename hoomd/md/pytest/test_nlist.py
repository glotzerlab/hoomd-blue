import numpy as np
import pytest
import random
from hoomd.md.nlist import Cell, Stencil, Tree


def _assert_nlist_params(nlist, param_dict):
    """Assert the params of the nlist are the same as in the dictionary."""
    for param in param_dict.keys():
        assert getattr(nlist, param) == param_dict[param]


@pytest.mark.parametrize("nlist_cls", [Cell, Tree, Stencil])
def test_common_params(nlist_cls):
    nlist = nlist_cls()
    default_params_dict = {
        "buffer": 0.4,
        "exclusions": ('bond',),
        "rebuild_check_delay": 1,
        "diameter_shift": False,
        "check_dist": True,
        "max_diameter": 1.0
    }
    _assert_nlist_params(nlist, default_params_dict)
    new_params_dict = {
        "buffer": np.random.uniform(5.0),
        "exclusions": random.sample(['bond','1-4', 'angle', 'dihedral',
                                     'special_pair', 'body', '1-3', 'constraint'],
                                    np.random.randint(9)),
        "rebuild_check_delay": np.random.randint(8),
        "diameter_shift": True,
        "check_dist": False,
        "max_diameter": np.random.uniform(10.3)
    }
    for param in new_params_dict.keys():
        setattr(nlist, param, new_params_dict[param])
    _assert_nlist_params(nlist, new_params_dict)


def test_cell_specific_params():
    nlist = Cell()
    _assert_nlist_params(nlist, dict(deterministic=False))
    nlist.deterministic = True
    _assert_nlist_params(nlist, dict(deterministic=True))


def test_stencil_specific_params():
    nlist = Stencil()
    _assert_nlist_params(nlist, dict(deterministic=False, cell_width=None))
    nlist.deterministic = True
    x = np.random.uniform(25.5)
    nlist.cell_width = x
    _assert_nlist_params(nlist, dict(deterministic=True, cell_width=x))


@pytest.mark.parametrize("nlist_cls", [Cell, Tree, Stencil])
def test_simple_simulation(nlist_cls, device, lattice_snapshot_factory):
    pass
