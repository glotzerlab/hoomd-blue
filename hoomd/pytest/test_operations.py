# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd


class FakeIntegrator(hoomd.operation.Integrator):
    pass


def test_len():
    operations = hoomd.Operations()
    # ParticleSorter automatically added
    assert len(operations) == 1
    operations.tuners.clear()
    assert len(operations) == 0

    operations.integrator = FakeIntegrator()
    operations.updaters.append(
        hoomd.update.FilterUpdater(1, [hoomd.filter.Type(["A"])]))
    operations.writers.append(hoomd.write.GSD(1, "filename.gsd"))

    assert len(operations) == 3


def test_iter():
    operations = hoomd.Operations()
    # ParticleSorter automatically added
    assert len(list(operations)) == 1

    operations.updaters.append(
        hoomd.update.FilterUpdater(1, [hoomd.filter.Type(["A"])]))
    operations.writers.append(hoomd.write.GSD(1, "filename.gsd"))

    expected_list = (operations._tuners[:] + operations._updaters[:]
                     + operations._writers[:])
    assert list(operations) == expected_list

    operations.integrator = FakeIntegrator()
    expected_list.insert(2, operations.integrator)
    assert list(operations) == expected_list
