# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from math import isclose

import numpy as np
import pytest
from pytest import fixture

from hoomd.box import Box


@fixture
def box_dict():
    return dict(Lx=1, Ly=2, Lz=3, xy=1, xz=2, yz=3)


def test_base_constructor(box_dict):
    box = Box(**box_dict)
    for key in box_dict:
        assert getattr(box, key) == box_dict[key]


@fixture
def base_box(box_dict):
    return Box(**box_dict)


def test_cpp_python_correspondence(base_box):
    cpp_obj = base_box._cpp_obj
    cpp_L = cpp_obj.getL()
    assert base_box.Lx == cpp_L.x and base_box.Ly == cpp_L.y \
        and base_box.Lz == cpp_L.z
    assert base_box.xy == cpp_obj.getTiltFactorXY()
    assert base_box.xz == cpp_obj.getTiltFactorXZ()
    assert base_box.yz == cpp_obj.getTiltFactorYZ()


def test_setting_lengths(base_box):
    for attr in ['Lx', 'Ly', 'Lz']:
        for L in np.linspace(1, 100, 10):
            setattr(base_box, attr, L)
            assert getattr(base_box, attr) == L
    for L in np.linspace(1, 100, 10):
        base_box.L = L
        assert all(base_box.L == L)
    base_box.L = [3, 2, 1]
    assert all(base_box.L == [3, 2, 1])


def test_setting_tilts(base_box):
    for attr in ['xy', 'xz', 'yz']:
        for tilt in np.linspace(1, 100, 10):
            setattr(base_box, attr, tilt)
            assert getattr(base_box, attr) == tilt
    for tilt in np.linspace(1, 100, 10):
        base_box.tilts = tilt
        assert all(base_box.tilts == tilt)
    base_box.tilts = [3, 2, 1]
    assert all(base_box.tilts == [3, 2, 1])


def test_is2D(base_box):  # noqa: N802 - allow function name
    base_box.Lz = 0
    assert base_box.is2D
    for L in np.linspace(1, 100, 10):
        base_box.Lz = L
        assert not base_box.is2D


def test_dimensions(base_box):
    base_box.Lz = 0
    assert base_box.dimensions == 2
    for L in np.linspace(1, 100, 10):
        base_box.Lz = L
        assert base_box.dimensions == 3


def get_aspect(L):
    return np.array([L[0] / L[1], L[0] / L[2], L[1] / L[2]])


def test_scale(base_box):
    aspect = get_aspect(base_box.L)
    for s in np.linspace(0.5, 1.5, 10):
        prev_vol = base_box.volume
        base_box.scale(s)
        assert np.allclose(aspect, get_aspect(base_box.L))
        assert not isclose(prev_vol, base_box.volume)

    L = base_box.L
    s = np.array([1, 0.75, 0.5])
    base_box.scale(s)
    assert np.allclose(aspect * get_aspect(s), get_aspect(base_box.L))
    assert np.allclose(base_box.L, L * s)


def test_volume(base_box):
    assert isclose(base_box.volume, np.prod(base_box.L))
    for L in np.linspace(1, 10, 10):
        box = Box.cube(L)
        assert isclose(box.volume, L**3)
        box = Box(L, L + 1, L + 2)
        assert isclose(box.volume, L * (L + 1) * (L + 2))


def test_volume_setting(base_box):
    aspect = get_aspect(base_box.L)
    for v in np.linspace(1, 100, 10):
        base_box.volume = v
        assert np.allclose(aspect, get_aspect(base_box.L))
        assert isclose(base_box.volume, v)


def test_periodic(base_box):
    assert all(base_box.periodic)


@fixture
def expected_matrix(box_dict):
    return np.array([
        [
            box_dict['Lx'], box_dict['Ly'] * box_dict['xy'],
            box_dict['Lz'] * box_dict['xz']
        ],
        [0, box_dict['Ly'], box_dict['Lz'] * box_dict['yz']],
        [0, 0, box_dict['Lz']],
    ])


def test_matrix(base_box, expected_matrix):
    assert np.allclose(base_box.to_matrix(), expected_matrix)
    base_box.xy *= 2
    assert isclose(base_box.to_matrix()[0, 1], 2 * expected_matrix[0, 1])
    base_box.yz *= 0.5
    assert isclose(base_box.to_matrix()[1, 2], 0.5 * expected_matrix[1, 2])
    base_box.Lx *= 3
    assert isclose(base_box.to_matrix()[0, 0], 3 * expected_matrix[0, 0])


@fixture
def new_box_matrix_dict():
    Lx, Ly, Lz = 2, 4, 8
    xy, xz, yz = 1, 3, 5
    new_box_matrix = np.array([[Lx, Ly * xy, Lz * xz], [0, Ly, Lz * yz],
                               [0, 0, Lz]])
    return dict(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz, matrix=new_box_matrix)


def test_cube():
    for L in np.linspace(1, 100, 10):
        box = Box.cube(L)
        assert all(box.L == L)
        assert box.Lx == box.Ly == box.Lz == L


def test_square():
    for L in np.linspace(1, 100, 10):
        box = Box.square(L)
        assert all(box.L == [L, L, 0])
        assert box.Lx == box.Ly == L and box.Lz == 0


def test_from_matrix(new_box_matrix_dict):
    box = Box.from_matrix(new_box_matrix_dict['matrix'])
    assert np.allclose(new_box_matrix_dict['matrix'], box.to_matrix())
    assert np.allclose(box.L, [
        new_box_matrix_dict['Lx'], new_box_matrix_dict['Ly'],
        new_box_matrix_dict['Lz']
    ])
    assert np.allclose(box.tilts, [
        new_box_matrix_dict['xy'], new_box_matrix_dict['xz'],
        new_box_matrix_dict['yz']
    ])


@pytest.mark.parametrize("theta",
                         [np.pi, np.pi / 2, np.pi / 3, np.pi / 4, np.pi * 1.23])
def test_from_basis_vectors_non_triangular(theta):
    box_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    box, rotation = Box.from_basis_vectors(box_matrix.T)
    assert np.allclose([box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz],
                       [1, 1, 1, 0, 0, 0],
                       atol=1e-6)
    rotated_matrix = box.to_matrix()
    rotated_points = rotation @ box_matrix
    assert np.allclose(rotated_matrix, rotated_points)


def test_from_matrix_two_dimensional():
    box_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    box, _ = Box.from_basis_vectors(box_matrix)
    assert box.is2D and box.dimensions == 2


@pytest.mark.parametrize("theta",
                         [np.pi, np.pi / 2, np.pi / 3, np.pi / 4, np.pi * 1.23])
def test_rotation_matrix_from_basis_vectors_two_dimensional(theta):
    box_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0], [0, 0, 0]])
    box, rotation = Box.from_basis_vectors(box_matrix.T)
    rotated_matrix = box.to_matrix()
    rotated_points = rotation @ box_matrix
    assert np.allclose(rotated_matrix, rotated_points)
    assert box.is2D and box.dimensions == 2


def test_invalid_from_basis_vectors_two_dimensional():
    box_matrix = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])
    import pytest
    with pytest.raises(ValueError):
        Box.from_basis_vectors(box_matrix)


def test_eq(base_box, box_dict):
    box2 = Box(**box_dict)
    assert base_box == box2
    box2.Lx = 2
    assert not base_box == box2


def test_neq(base_box, box_dict):
    box2 = Box(**box_dict)
    assert not base_box != box2
    box2.Lx = 2
    assert base_box != box2
