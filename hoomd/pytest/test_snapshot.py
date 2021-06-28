from hoomd.snapshot import Snapshot
import numpy
import pytest
from hoomd.pytest.test_simulation import make_gsd_snapshot
try:
    import gsd.hoomd  # noqa: F401 - need to know if the import fails
    skip_gsd = False
except ImportError:
    skip_gsd = True

skip_gsd = pytest.mark.skipif(skip_gsd,
                              reason="gsd Python package was not found.")


def assert_equivalent_snapshots(gsd_snap, hoomd_snap):
    """This is the same as the function in test_simulation.

    Except that if a prop is None in the gsd_snap, it instead checks that the
    array or list is empty in the hoomd_snap additionally it handles the
    differences in the way the two snapshots handle the dimensions of empty
    boxes. This function returns ``True`` when not on the root rank.
    """
    if not hoomd_snap.communicator.rank == 0:
        return True
    for attr in dir(hoomd_snap):
        if attr[0] == '_' or attr in ['exists', 'replicate', 'communicator']:
            continue
        for prop in dir(getattr(hoomd_snap, attr)):
            if prop[0] == '_':
                continue
            elif prop == 'types':
                x = getattr(getattr(gsd_snap, attr), prop)
                y = getattr(getattr(hoomd_snap, attr), prop)
                if x is None:
                    assert y == []
                else:
                    assert x == y
            elif prop == 'dimensions':
                x = getattr(getattr(gsd_snap, attr), prop)
                y = getattr(getattr(hoomd_snap, attr), prop)
                x_box = getattr(getattr(gsd_snap, attr), 'box')
                if x_box is None or x_box.all() == 0:
                    # if the box is all zeros, the dimensions won't match
                    # hoomd dimensions will be 2 and gsd will be 3
                    continue
                elif x is None:
                    assert y == []
                else:
                    assert x == y
            elif prop == 'acceleration' or prop == 'is_accel_set':
                continue
            else:
                x = getattr(getattr(gsd_snap, attr), prop)
                y = getattr(getattr(hoomd_snap, attr), prop)
                if x is None:
                    numpy.testing.assert_allclose(numpy.zeros_like(y), y)
                else:
                    numpy.testing.assert_allclose(x, y)


@pytest.fixture(scope='function')
def s():
    return Snapshot()


def test_empty_snapshot(s):
    if s.communicator.rank == 0:
        numpy.testing.assert_allclose(s.configuration.box, [0, 0, 0, 0, 0, 0],
                                      atol=1e-7)
        assert s.configuration.dimensions == 3

        assert s.particles.N == 0
        assert len(s.particles.position) == 0
        assert len(s.particles.velocity) == 0
        assert len(s.particles.acceleration) == 0
        assert len(s.particles.typeid) == 0
        assert len(s.particles.mass) == 0
        assert len(s.particles.charge) == 0
        assert len(s.particles.diameter) == 0
        assert len(s.particles.image) == 0
        assert len(s.particles.body) == 0
        assert len(s.particles.orientation) == 0
        assert len(s.particles.moment_inertia) == 0
        assert len(s.particles.angmom) == 0
        assert len(s.particles.types) == 0

        assert s.bonds.N == 0
        assert len(s.bonds.types) == 0
        assert len(s.bonds.typeid) == 0
        assert len(s.bonds.group) == 0

        assert s.angles.N == 0
        assert len(s.angles.types) == 0
        assert len(s.angles.typeid) == 0
        assert len(s.angles.group) == 0

        assert s.dihedrals.N == 0
        assert len(s.dihedrals.types) == 0
        assert len(s.dihedrals.typeid) == 0
        assert len(s.dihedrals.group) == 0

        assert s.impropers.N == 0
        assert len(s.impropers.types) == 0
        assert len(s.impropers.typeid) == 0
        assert len(s.impropers.group) == 0

        assert s.pairs.N == 0
        assert len(s.pairs.types) == 0
        assert len(s.pairs.typeid) == 0
        assert len(s.pairs.group) == 0

        assert s.constraints.N == 0
        assert len(s.constraints.value) == 0
        assert len(s.constraints.group) == 0


def test_configuration(s):
    if s.communicator.rank == 0:
        s.configuration.box = [10, 12, 7, 0.1, 0.4, 0.2]
        numpy.testing.assert_allclose(s.configuration.box,
                                      [10, 12, 7, 0.1, 0.4, 0.2])

        with pytest.raises(AttributeError):
            s.configuration.dimensions = 2
        assert s.configuration.dimensions == 3


def test_wrap(s):
    def generateOutside(box, inside, multipliers):
        a = numpy.array([box[0], 0, 0])
        b = numpy.array([box[1]*box[3], box[1], 0])
        c = numpy.array([box[2]*box[4], box[2]*box[5], box[2]])
        out = numpy.zeros((len(inside),len(multipliers),3))
        ins = numpy.zeros_like(out)
        for i,point in enumerate(inside):
            for j,f in enumerate(multipliers):
                out[i,j,:] = point+ a*f[0] + b*f[1] + c*f[2]
                ins[i,j,:] = point
        return out.reshape((-1,3)), ins.reshape((-1,3))

    if s.communicator.rank == 0:
        s.configuration.box = [1, 1, 1, 0, 0, 0]
        s.particles.N = 4
        s.particles.position[:] = [
            [1.5, 2, 1.8],  # test all directions
            [-1.0, -0.7, -10.1],  # test negative directions
            [0.3, 0.2, -0.1],  # test no wrap
            [2.5, 100.2, -100.2]
        ]  # test multiple wrap
        s.wrap()
        numpy.testing.assert_allclose(s.particles.position,
                                      [[-0.5, 0.0, -0.2], [0.0, 0.3, -0.1],
                                       [0.3, 0.2, -0.1], [-0.5, 0.2, -0.2]])
        # triclinic box
        s.configuration.box = [10, 12, 7, 0.1, 0.4, 0.2]
        s.particles.N = 4
        s.particles.position[:] = [[-4, 0, 0], [2, 3, 1], [2, -8, 9],
                                   [100, 200, -100]]
        s.wrap()
        numpy.testing.assert_allclose(
            s.particles.position,
            [[-4, 0, 0], [2, 3, 1],
             [0.4, 2.6, 2], [-2.4, 3.6, -2]])

        # 2D box
        b = [5, 11, 0, 0, 0, 0]
        inside = [[1, 0, 0], [2, 5, 0], [2, 3, 0], [0, 5, 0]]
        multiples = [[1,0,0], [0,0,0], [-1,0,0]]
        outs, ins = generateOutside(b, inside, multiples)

        s.configuration.box = b
        s.particles.N = len(ins)
        s.particles.position[:] = outs
        s.wrap()
        numpy.testing.assert_allclose(s.particles.position, ins)

        # tetragonal box
        s.configuration.box = [7, 7, 4, 0, 0, 0]
        s.particles.N = 4
        s.particles.position[:] = [[-8, 0, 0], [-4, -8, -7], [0, 2, 1],
                                   [50, 100, -300]]
        s.wrap()
        numpy.testing.assert_allclose(
            s.particles.position,
            [[-1, 0, 0], [3, -1, 1], [0, 2, 1], [1, 2, 0]])
        # orthorhombic box
        s.configuration.box = [8, 6, 4, 0, 0, 0]
        s.particles.N = 4
        s.particles.position[:] = [[-4, 3, 2], [0, -6, 10], [2, 0, -1],
                                   [-200, -200, -100]]
        s.wrap()
        numpy.testing.assert_allclose(
            s.particles.position,
            [[-4, -3, -2], [0, 0, -2], [2, 0, -1], [0, -2, 0]])
        # monoclinic box
        s.configuration.box = [7, 4, 8, 0, 0.25, 0]
        s.particles.N = 4
        s.particles.position[:] = [[-4, 9, -9], [0, 4, 13], [2, 1, 1],
                                   [-200, -100, -100]]
        s.wrap()
        numpy.testing.assert_allclose(
            s.particles.position,
            [[-2, 1, -1], [-4, 0, -3], [2, 1, 1], [-1, 0, -4]])


def test_particles(s):
    if s.communicator.rank == 0:
        s.particles.N = 5

        assert s.particles.N == 5
        assert len(s.particles.position) == 5
        assert len(s.particles.velocity) == 5
        assert len(s.particles.acceleration) == 5
        assert len(s.particles.typeid) == 5
        assert len(s.particles.mass) == 5
        assert len(s.particles.charge) == 5
        assert len(s.particles.diameter) == 5
        assert len(s.particles.image) == 5
        assert len(s.particles.body) == 5
        assert len(s.particles.orientation) == 5
        assert len(s.particles.moment_inertia) == 5
        assert len(s.particles.angmom) == 5

        s.particles.types = ['A', 'B']
        assert s.particles.types == ['A', 'B']

        assert s.particles.position.dtype == numpy.float64
        assert s.particles.position.shape == (5, 3)
        assert s.particles.velocity.dtype == numpy.float64
        assert s.particles.velocity.shape == (5, 3)
        assert s.particles.acceleration.dtype == numpy.float64
        assert s.particles.acceleration.shape == (5, 3)
        assert s.particles.typeid.dtype == numpy.uint32
        assert s.particles.typeid.shape == (5,)
        assert s.particles.mass.dtype == numpy.float64
        assert s.particles.mass.shape == (5,)
        assert s.particles.charge.dtype == numpy.float64
        assert s.particles.charge.shape == (5,)
        assert s.particles.diameter.dtype == numpy.float64
        assert s.particles.diameter.shape == (5,)
        assert s.particles.image.dtype == numpy.int32
        assert s.particles.image.shape == (5, 3)
        assert s.particles.body.dtype == numpy.int32
        assert s.particles.body.shape == (5,)
        assert s.particles.orientation.dtype == numpy.float64
        assert s.particles.orientation.shape == (5, 4)
        assert s.particles.moment_inertia.dtype == numpy.float64
        assert s.particles.moment_inertia.shape == (5, 3)
        assert s.particles.angmom.dtype == numpy.float64
        assert s.particles.angmom.shape == (5, 4)


def test_bonds(s):
    if s.communicator.rank == 0:
        s.bonds.N = 3

        assert s.bonds.N == 3
        assert len(s.bonds.typeid) == 3
        assert len(s.bonds.group) == 3

        s.bonds.types = ['A', 'B']
        assert s.bonds.types == ['A', 'B']

        assert s.bonds.typeid.shape == (3,)
        assert s.bonds.typeid.dtype == numpy.uint32
        assert s.bonds.group.shape == (3, 2)
        assert s.bonds.group.dtype == numpy.uint32


def test_angles(s):
    if s.communicator.rank == 0:
        s.angles.N = 3

        assert s.angles.N == 3
        assert len(s.angles.typeid) == 3
        assert len(s.angles.group) == 3

        s.angles.types = ['A', 'B']
        assert s.angles.types == ['A', 'B']

        assert s.angles.typeid.shape == (3,)
        assert s.angles.typeid.dtype == numpy.uint32
        assert s.angles.group.shape == (3, 3)
        assert s.angles.group.dtype == numpy.uint32


def test_dihedrals(s):
    if s.communicator.rank == 0:
        s.dihedrals.N = 3

        assert s.dihedrals.N == 3
        assert len(s.dihedrals.typeid) == 3
        assert len(s.dihedrals.group) == 3

        s.dihedrals.types = ['A', 'B']
        assert s.dihedrals.types == ['A', 'B']

        assert s.dihedrals.typeid.shape == (3,)
        assert s.dihedrals.typeid.dtype == numpy.uint32
        assert s.dihedrals.group.shape == (3, 4)
        assert s.dihedrals.group.dtype == numpy.uint32


def test_impropers(s):
    if s.communicator.rank == 0:
        s.impropers.N = 3

        assert s.impropers.N == 3
        assert len(s.impropers.typeid) == 3
        assert len(s.impropers.group) == 3

        s.impropers.types = ['A', 'B']
        assert s.impropers.types == ['A', 'B']

        assert s.impropers.typeid.shape == (3,)
        assert s.impropers.typeid.dtype == numpy.uint32
        assert s.impropers.group.shape == (3, 4)
        assert s.impropers.group.dtype == numpy.uint32


def test_pairs(s):
    if s.communicator.rank == 0:
        s.pairs.N = 3

        assert s.pairs.N == 3
        assert len(s.pairs.typeid) == 3
        assert len(s.pairs.group) == 3

        s.pairs.types = ['A', 'B']
        assert s.pairs.types == ['A', 'B']

        assert s.pairs.typeid.shape == (3,)
        assert s.pairs.typeid.dtype == numpy.uint32
        assert s.pairs.group.shape == (3, 2)
        assert s.pairs.group.dtype == numpy.uint32


def test_constraints(s):
    if s.communicator.rank == 0:
        s.constraints.N = 3

        assert s.constraints.N == 3
        assert len(s.constraints.value) == 3
        assert len(s.constraints.group) == 3

        assert s.constraints.value.shape == (3,)
        assert s.constraints.value.dtype == numpy.float64
        assert s.constraints.group.shape == (3, 2)
        assert s.constraints.group.dtype == numpy.uint32


@skip_gsd
def test_from_gsd_snapshot_empty(s, device):
    gsd_snap = make_gsd_snapshot(s)
    hoomd_snap = Snapshot.from_gsd_snapshot(gsd_snap, device.communicator)
    assert_equivalent_snapshots(gsd_snap, hoomd_snap)


@skip_gsd
def test_from_gsd_snapshot_populated(s, device):
    if s.communicator.rank == 0:
        s.configuration.box = [10, 12, 7, 0.1, 0.4, 0.2]
        for section in ('particles', 'bonds', 'angles', 'dihedrals',
                        'impropers', 'pairs'):
            setattr(getattr(s, section), 'N', 5)
            setattr(getattr(s, section), 'types', ['A', 'B'])

        for prop in ('angmom', 'body', 'charge', 'diameter', 'image', 'mass',
                     'moment_inertia', 'orientation', 'position', 'typeid',
                     'velocity'):
            attr = getattr(s.particles, prop)
            if attr.dtype == numpy.float64:
                attr[:] = numpy.random.rand(*attr.shape)
            else:
                attr[:] = numpy.random.randint(3, size=attr.shape)

        for section in ('bonds', 'angles', 'dihedrals', 'impropers', 'pairs'):
            for prop in ('group', 'typeid'):
                attr = getattr(getattr(s, section), prop)
                attr[:] = numpy.random.randint(3, size=attr.shape)

        s.constraints.N = 3
        for prop in ('group', 'value'):
            attr = getattr(s.constraints, prop)
            if attr.dtype == numpy.float64:
                attr[:] = numpy.random.rand(*attr.shape)
            else:
                attr[:] = numpy.random.randint(3, size=attr.shape)

    gsd_snap = make_gsd_snapshot(s)
    hoomd_snap = Snapshot.from_gsd_snapshot(gsd_snap, device.communicator)
    assert_equivalent_snapshots(gsd_snap, hoomd_snap)


def test_invalid_particle_typeids(simulation_factory, lattice_snapshot_factory):
    """Test that using invalid particle typeids raises an error."""
    snap = lattice_snapshot_factory(particle_types=['A', 'B'])

    # assign invalid type ids
    if snap.communicator.rank == 0:
        snap.particles.typeid[:] = 2

    with pytest.raises(RuntimeError):
        simulation_factory(snap)


def test_no_particle_types(simulation_factory, lattice_snapshot_factory):
    """Test that initialization fails when there are no types."""
    snap = lattice_snapshot_factory(particle_types=[])

    with pytest.raises(RuntimeError):
        simulation_factory(snap)


def test_zero_particle_system(simulation_factory, lattice_snapshot_factory):
    """Test that zero particle systems can be initialized with no types."""
    snap = lattice_snapshot_factory(particle_types=[], n=0)

    simulation_factory(snap)


@pytest.mark.parametrize("group_name,group_size", [
    ("bonds", 2),
    ("angles", 3),
    ("dihedrals", 4),
    ("impropers", 4),
    ("pairs", 2),
])
def test_invalid_bond_typeids(
    group_name,
    group_size,
    simulation_factory,
    lattice_snapshot_factory,
):
    """Test that using invalid bond typeids raises an error."""
    snap = lattice_snapshot_factory()

    # assign invalid type ids
    if snap.communicator.rank == 0:
        group = getattr(snap, group_name)
        group.types = ['A']
        group.N = 1
        group.group[0] = range(group_size)
        group.typeid[:] = 2

    with pytest.raises(RuntimeError):
        simulation_factory(snap)

    # test that 0 types is allowed when there are 0 items in the group
    snap = lattice_snapshot_factory()

    # assign invalid type ids
    if snap.communicator.rank == 0:
        group = getattr(snap, group_name)
        group.types = []
        group.N = 0

    simulation_factory(snap)
