# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

from hoomd.tune import ManualTuneDefinition
from hoomd.tune import ScaleSolver, SecantSolver


@pytest.fixture(params=[ScaleSolver(), SecantSolver()],
                ids=lambda solver: solver.__class__.__name__)
def solver(request):
    return request.param


@pytest.fixture
def equation_definition():
    """Evaluate: x^2 - 1, x = (1, -1)."""
    equation = dict(x=4)
    equation['y'] = lambda: equation['x']**2
    return ManualTuneDefinition(get_x=lambda: equation['x'],
                                set_x=lambda x: equation.__setitem__('x', x),
                                get_y=lambda: equation['y'](),
                                target=1)


class TestRootSolvers:

    def test_solving(self, solver, equation_definition):
        cnt = 0
        complete = False
        while not complete:
            complete = solver.solve([equation_definition])
            cnt += 1
            if cnt >= 500:
                err = equation_definition.y - equation_definition.target
                raise RuntimeError(
                    "Expected conversion earlier: err={}.".format(err))
        err = equation_definition.y - equation_definition.target
        solutions = (-1, 1)
        assert abs(err) <= solver.tol
        assert any(
            abs(equation_definition.x - sol) <= 1e-3 for sol in solutions)
