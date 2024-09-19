# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from __future__ import print_function
from __future__ import division
import hoomd
import pytest
import numpy
import hoomd.hpmc.pytest.conftest
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check

# TODO: reimplement test without LLVM
llvm_disabled = True


def test_before_attaching():
    xmax = 0.02
    dx = 1e-4
    sdf = hoomd.hpmc.compute.SDF(xmax=xmax, dx=dx)
    assert numpy.isclose(sdf.xmax, xmax)
    assert numpy.isclose(sdf.dx, dx)

    xmax = 0.04
    dx = 2e-4
    sdf.xmax = xmax
    sdf.dx = dx
    assert numpy.isclose(sdf.xmax, xmax)
    assert numpy.isclose(sdf.dx, dx)

    with pytest.raises(hoomd.error.DataAccessError):
        sdf.sdf_compression
    with pytest.raises(hoomd.error.DataAccessError):
        sdf.sdf_expansion
    with pytest.raises(hoomd.error.DataAccessError):
        sdf.betaP


def test_after_attaching(valid_args, simulation_factory,
                         lattice_snapshot_factory):
    integrator, args, n_dimensions = valid_args
    snap = lattice_snapshot_factory(particle_types=['A'],
                                    dimensions=n_dimensions)
    sim = simulation_factory(snap)

    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"].to_base()
    mc = integrator()
    mc.shape["A"] = args
    sim.operations.add(mc)

    xmax = 0.02
    dx = 1e-4
    sdf = hoomd.hpmc.compute.SDF(xmax=xmax, dx=dx)

    sim.operations.add(sdf)
    assert len(sim.operations.computes) == 1
    sim.run(0)

    assert numpy.isclose(sdf.xmax, xmax)
    assert numpy.isclose(sdf.dx, dx)

    xmax = 0.04
    dx = 4e-4
    sdf.xmax = xmax
    sdf.dx = dx
    assert numpy.isclose(sdf.xmax, xmax)
    assert numpy.isclose(sdf.dx, dx)

    sim.run(1)

    x_compression = sdf.x_compression
    assert x_compression[0] == pytest.approx(dx / 2)
    assert x_compression[-1] == pytest.approx(xmax - dx / 2)

    x_expansion = sdf.x_expansion
    assert x_expansion[0] == pytest.approx(-xmax + dx / 2)
    assert x_expansion[-1] == pytest.approx(-dx / 2)

    sdf_compression = sdf.sdf_compression
    betaP = sdf.betaP

    if sdf_compression is not None:
        assert sim.device.communicator.rank == 0
        assert isinstance(sdf_compression, numpy.ndarray)
        assert len(sdf_compression) > 0
        assert isinstance(betaP, float)
        assert not numpy.isclose(betaP, 0)
    else:
        assert sim.device.communicator.rank > 0
        assert betaP is None

    # Regression test for array size mismatch bug:
    # https://github.com/glotzerlab/hoomd-blue/issues/1455

    # only test for 1 shape
    if not isinstance(mc, hoomd.hpmc.integrate.Sphere):
        return

    sdf.xmax = 0.02
    sdf.dx = 1e-5

    sim.run(1)
    sdf_compression = sdf.sdf_compression
    betaP = sdf.betaP

    if sdf_compression is not None:
        assert sim.device.communicator.rank == 0
        assert isinstance(sdf_compression, numpy.ndarray)
        assert len(sdf_compression) > 0
        assert isinstance(betaP, float)
        assert not numpy.isclose(betaP, 0)
    else:
        assert sim.device.communicator.rank > 0
        assert betaP is None


_avg = numpy.array([
    55.20126953, 54.89853516, 54.77910156, 54.56660156, 54.22255859,
    53.83935547, 53.77617188, 53.42109375, 53.05546875, 52.86376953,
    52.65576172, 52.21240234, 52.07402344, 51.88974609, 51.69990234,
    51.32099609, 51.09775391, 51.06533203, 50.61923828, 50.35566406,
    50.07197266, 49.92275391, 49.51914062, 49.39013672, 49.17597656,
    48.91982422, 48.64580078, 48.30712891, 48.12207031, 47.815625, 47.57744141,
    47.37099609, 47.14765625, 46.92382812, 46.6984375, 46.66943359, 46.18203125,
    45.95615234, 45.66650391, 45.52714844, 45.39951172, 45.04599609,
    44.90908203, 44.62197266, 44.37460937, 44.02998047, 43.84306641,
    43.53310547, 43.55, 43.29589844, 43.06054688, 42.85097656, 42.58837891,
    42.39326172, 42.21152344, 41.91777344, 41.71054687, 41.68232422,
    41.42177734, 41.08085938, 40.91435547, 40.76123047, 40.45380859, 40.178125,
    40.14853516, 39.81972656, 39.60585938, 39.44169922, 39.34179688,
    39.09541016, 38.78105469, 38.60087891, 38.56572266, 38.27158203,
    38.02011719, 37.865625, 37.77851562, 37.51113281, 37.25615234, 37.23857422,
    36.91757812, 36.68486328, 36.57675781, 36.39140625, 36.06240234,
    36.01962891, 35.8375, 35.51914062, 35.3640625, 35.29042969, 34.86337891,
    34.72460938, 34.73964844, 34.57871094, 34.32685547, 34.02607422,
    33.78271484, 33.82548828, 33.53808594, 33.40341797, 33.17861328,
    33.05439453, 32.80361328, 32.55478516, 32.53759766, 32.28447266,
    32.26513672, 32.05732422, 31.82294922, 31.83535156, 31.56376953,
    31.46337891, 31.27431641, 30.88310547, 30.85107422, 30.63320313,
    30.57822266, 30.28886719, 30.28183594, 30.05927734, 29.98896484, 29.690625,
    29.51816406, 29.40742188, 29.2328125, 29.19853516, 28.94599609, 28.80449219,
    28.47480469, 28.48476563, 28.31738281, 28.21455078, 28.00878906,
    27.90458984, 27.84970703, 27.54052734, 27.43818359, 27.31064453,
    27.12773437, 26.91464844, 26.84511719, 26.78701172, 26.53603516,
    26.39853516, 26.13779297, 26.16269531, 25.92138672, 25.80244141,
    25.75234375, 25.49384766, 25.37197266, 25.26962891, 25.14287109,
    24.87558594, 24.778125, 24.68320312, 24.65957031, 24.44404297, 24.31621094,
    24.203125, 24.12402344, 23.89628906, 23.76621094, 23.56923828, 23.38095703,
    23.32724609, 23.25498047, 23.09697266, 23.04716797, 22.90712891,
    22.68662109, 22.59970703, 22.54824219, 22.53632813, 22.29267578,
    22.08613281, 21.98398437, 21.89169922, 21.74550781, 21.75878906, 21.45625,
    21.37529297, 21.1890625, 21.18417969, 21.0671875, 20.95087891, 20.81650391,
    20.60390625, 20.66953125, 20.4640625, 20.47021484, 20.12988281, 20.17099609,
    20.05224609, 19.89619141, 19.80859375, 19.72558594, 19.64990234,
    19.43525391, 19.38203125
])

_err = numpy.array([
    1.21368492, 1.07520243, 1.22496485, 1.07203861, 1.31918198, 1.15482965,
    1.11606943, 1.12342247, 1.1214123, 1.2033176, 1.14923442, 1.11741796,
    1.08633901, 1.10809585, 1.13268611, 1.17159683, 1.12298656, 1.27754418,
    1.09430177, 1.08989947, 1.051715, 1.13990382, 1.16086636, 1.19538929,
    1.09450355, 1.10057404, 0.98204849, 1.02542969, 1.10736805, 1.18062055,
    1.12365972, 1.12265463, 1.06131492, 1.15169701, 1.13772836, 1.03968987,
    1.04348243, 1.00617502, 1.02450203, 1.08293272, 1.02187476, 1.00072731,
    1.0267637, 1.08289546, 1.03696814, 1.01035732, 1.05730499, 1.07088231,
    1.00528653, 0.9195167, 0.99235353, 1.00839744, 0.98700882, 0.87196929,
    1.00124084, 0.96481759, 0.9412312, 1.04691734, 0.92419062, 0.89478269,
    0.85106599, 1.0143535, 1.07011876, 0.88196475, 0.8708013, 0.91838154,
    0.9309356, 0.97521482, 0.94277816, 0.86336248, 0.8845162, 1.00421706,
    0.87940419, 0.85516477, 0.86071935, 0.96725404, 0.87175829, 0.86386878,
    0.96833751, 0.87554994, 0.8449041, 0.77404494, 0.92879454, 0.95780868,
    0.84341047, 0.88067771, 0.83393048, 0.94414754, 0.94671484, 0.84554255,
    0.8906436, 0.84538732, 0.78517686, 0.89134056, 0.78446042, 0.8952503,
    0.84624311, 0.79573064, 0.85422345, 0.88918562, 0.75531048, 0.82884413,
    0.83369698, 0.77627999, 0.84187759, 0.87986859, 0.86356705, 0.90929237,
    0.83017397, 0.86393341, 0.81426374, 0.80991068, 0.86676111, 0.75232448,
    0.8021119, 0.68794232, 0.69039919, 0.71421068, 0.77667793, 0.82113389,
    0.70256397, 0.83293526, 0.69512453, 0.75148262, 0.7407287, 0.74124134,
    0.77846167, 0.7941425, 0.81125561, 0.73334183, 0.76452184, 0.71159507,
    0.67302729, 0.66175046, 0.84778683, 0.66273563, 0.76777339, 0.71355888,
    0.74460445, 0.76623613, 0.63883733, 0.6887326, 0.74616778, 0.65223179,
    0.76358086, 0.68985286, 0.66273563, 0.72437662, 0.77382571, 0.66234322,
    0.74757211, 0.62809942, 0.75606851, 0.65375498, 0.65920693, 0.64767863,
    0.67683992, 0.63170556, 0.69891621, 0.70708048, 0.64583276, 0.73903135,
    0.60068155, 0.66055863, 0.69614341, 0.61515868, 0.63001311, 0.68602529,
    0.7014929, 0.61950453, 0.60049188, 0.6259654, 0.55819764, 0.65039367,
    0.67079534, 0.60552195, 0.64864663, 0.59901689, 0.65517427, 0.55348699,
    0.57578738, 0.6253923, 0.62679547, 0.61274744, 0.5681065, 0.6065114,
    0.61170127, 0.60009145, 0.61583989, 0.63889728, 0.66477228, 0.60133457,
    0.56484264, 0.5676353, 0.55359946, 0.59000379, 0.60483562, 0.57305916,
    0.57591598, 0.66462928
])


@pytest.mark.validate
def test_values(simulation_factory, lattice_snapshot_factory):
    n_particles_per_side = 32
    phi = 0.8  # packing fraction

    poly_A = 1  # assumes squares
    N = n_particles_per_side**2
    area = N * poly_A / phi
    L = numpy.sqrt(area)
    a = L / n_particles_per_side

    snap = lattice_snapshot_factory(dimensions=2, n=n_particles_per_side, a=a)
    sim = simulation_factory(snap)
    sim.seed = 10

    mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=0.1)
    mc.shape["A"] = {
        'vertices': [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    }
    sim.operations.add(mc)

    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)
    sim.operations.add(sdf)

    sdf_log = hoomd.conftest.ListWriter(sdf, 'sdf_compression')
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=sdf_log,
                                 trigger=hoomd.trigger.Periodic(10)))

    sim.run(6000)

    sdf_data = numpy.asarray(sdf_log.data)
    if sim.device.communicator.rank == 0:
        # skip the first frame in averaging, then check that all values are
        # within 3 error bars of the reference avg. This seems sufficient to get
        # good test results even with different seeds or GPU runs
        v = numpy.mean(sdf_data[1:, :], axis=0)
        invalid = numpy.abs(_avg - v) > (8 * _err)
        assert numpy.sum(invalid) == 0


@pytest.mark.skipif(llvm_disabled, reason='LLVM not enabled')
@pytest.mark.cpu  # SDF runs on the CPU only, no need to test on the GPU
def test_linear_search_path(simulation_factory, two_particle_snapshot_factory):
    """Test that adding patches changes the pressure calculation.

    The system and sdf compute here are constructed such that the first search
    bin creates a soft overlap and the second one creates a hard overlap. We
    test that the first overlap is found in the 2nd bin before adding patches
    and that an overlap is found in the first bin after adding patches. We also
    test that the SDF values are as expected: (1-exp(U-U_new)) * sdf.dx.
    """
    xmax = 0.02
    dx = 1e-3
    r_core = 0.5  # radius of hard core
    r_patch = 0.5001  # if r_ij < 2*r_patch, particles interact
    epsilon = 2.0  # strength of square well attraction
    sim = simulation_factory(two_particle_snapshot_factory(d=1.001101081081081))
    sim.seed = 0
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.0)
    mc.shape['A'] = {'diameter': 2 * r_core}
    sim.operations.add(mc)

    # sdf compute
    sdf = hoomd.hpmc.compute.SDF(xmax=xmax, dx=dx)
    sim.operations.add(sdf)

    # confirm that there is a hard overlap in the 2nd bin when there is no pair
    # potential and that the SDF is zero everywhere else
    sim.run(0)
    norm_factor = 1 / sdf.dx
    sdf_result = sdf.sdf_compression
    if sim.device.communicator.rank == 0:
        assert (sdf_result[1] == norm_factor)
        assert (numpy.count_nonzero(sdf_result) == 1)

    # add pair potential
    square_well = r'''float rsq = dot(r_ij, r_ij);
                    float rcut = param_array[1];
                    if (rsq > 2*rcut)
                        return 0.0f;
                    else
                        return -param_array[0];'''
    patch = hoomd.hpmc.pair.user.CPPPotential(r_cut=2.5 * r_patch,
                                              code=square_well,
                                              param_array=[epsilon, r_patch])
    mc.pair_potential = patch

    # for a soft overlap with a negative change in energy, the weight of the
    # overlap is zero, so we still expect all zeros in the sdf array
    sim.run(1)
    neg_mayerF = 1 - numpy.exp(epsilon)
    sdf_result = sdf.sdf_compression
    if sim.device.communicator.rank == 0:
        assert (numpy.count_nonzero(sdf_result) == 0)

    # change sign of epsilon, so that now there is a positive energy soft
    # overlap in bin 0
    epsilon *= -1
    patch.param_array[0] = epsilon
    sim.run(1)
    neg_mayerF = 1 - numpy.exp(epsilon)
    sdf_result = sdf.sdf_compression
    if sim.device.communicator.rank == 0:
        assert (numpy.count_nonzero(sdf_result) == 1)
        assert (sdf_result[0] == neg_mayerF * norm_factor)

    # extend patches so that there is a soft overlap with positive energy
    # in the configuration and there will be a change in energy on expansions
    # the change in energy is < 0 so the weight should be 0 and
    # sdf_expansion should be all zeros
    patch.param_array[1] += 2 * dx
    sim.run(1)
    sdf_result = sdf.sdf_expansion
    if sim.device.communicator.rank == 0:
        assert (numpy.count_nonzero(sdf_result) == 0)

    # change sign of epsilon so now the change in energy on expansion is
    # positive and the weight is nonzero and one of the sdf_expansion counts
    # is nonzero
    epsilon *= -1
    patch.param_array[0] = epsilon
    sim.run(1)
    neg_mayerF = 1 - numpy.exp(-epsilon)
    sdf_result = sdf.sdf_expansion
    if sim.device.communicator.rank == 0:
        assert (numpy.count_nonzero(sdf_result) == 1)
        assert (sdf_result[-1] == neg_mayerF * norm_factor)


@pytest.mark.cpu
@pytest.mark.serial
def test_sdf_expansion(simulation_factory, two_particle_snapshot_factory):
    """Test that sdf_expansion registers counts."""
    xmax = 0.02
    dx = 1e-3

    # Place two tetrominoes pieces offset so they interlock and will touch when
    # moved apart.
    snapshot = two_particle_snapshot_factory(dimensions=2)
    if snapshot.communicator.rank == 0:
        snapshot.particles.position[0] = [0, 0, 0]
        snapshot.particles.position[1] = [2 - dx / 4, 0.5, 0]
        snapshot.particles.orientation[1] = [0, 0, 0, -1]

    sim = simulation_factory(snapshot)
    mc = hoomd.hpmc.integrate.SimplePolygon()
    mc.shape['A'] = dict(vertices=[
        (-2, 1),
        (-2, -1),
        (2, -1),
        (2, 1),
        (1, 1),
        (1, -0.9),
        (-1, -0.9),
        (-1, 1),
    ])
    sim.operations.add(mc)

    # sdf compute
    sdf = hoomd.hpmc.compute.SDF(xmax=xmax, dx=dx)
    sim.operations.add(sdf)

    sim.run(0)

    # confirm no overlaps
    assert mc.overlaps == 0

    # confirm that there are no hits in the compression sdf
    assert numpy.count_nonzero(sdf.sdf_compression) == 0

    # confirm that there is one hit in the expansion sdf
    sdf_expansion = sdf.sdf_expansion
    assert numpy.count_nonzero(sdf_expansion) == 1
    assert sdf_expansion[-1] != 0


def test_logging():
    logging_check(
        hoomd.hpmc.compute.SDF, ('hpmc', 'compute'), {
            'betaP': {
                'category': LoggerCategories.scalar,
                'default': True
            },
            'sdf_compression': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'sdf_expansion': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'x_compression': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'x_expansion': {
                'category': LoggerCategories.sequence,
                'default': True
            },
        })
