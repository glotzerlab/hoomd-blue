# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

from hoomd.tune import ManualTuneDefinition
from hoomd.tune import (ScaleSolver, SecantSolver, GradientDescent,
                        GridOptimizer)
import hoomd.variant


class SolverTestBase:
    Y_TOL = 1e-3
    X_TOL = 1e-3

    def test_solving(self, solver, equation):
        cnt = 0
        complete = False
        while not complete:
            complete = solver.solve([equation])
            print(equation.x)
            cnt += 1
            if cnt >= 500:
                err = self.get_y_error_mag(equation)
                raise RuntimeError(
                    "Expected conversion earlier: err={}.".format(err))
        err = equation.y - equation.target
        assert self.get_y_error_mag(equation) <= getattr(
            solver, "tol", self.Y_TOL)
        assert self.get_x_error_mag(equation) <= self.X_TOL

    def get_x_error_mag(self, equation):
        return min(abs(equation.x - sol) for sol in self.SOLUTIONS)

    def get_y_error_mag(self, equation):
        return abs(equation.y - equation.target)


class TestRootSolvers(SolverTestBase):
    SOLUTIONS = (-1, 1)

    @pytest.fixture(params=[ScaleSolver(), SecantSolver()],
                    ids=lambda solver: solver.__class__.__name__)
    def solver(self, request):
        return request.param

    @pytest.fixture
    def equation(self):
        """Evaluate: x^2 - 1, x = (1, -1)."""
        equation = dict(x=4)
        equation['y'] = lambda: equation['x']**2
        return ManualTuneDefinition(
            get_x=lambda: equation['x'],
            set_x=lambda x: equation.__setitem__('x', x),
            get_y=lambda: equation['y'](),
            target=1)


class TestOptimizers(SolverTestBase):
    SOLUTIONS = (2,)
    Y_TOL = 1e-2
    X_TOL = 2e-3

    @pytest.fixture(params=[GradientDescent(),
                            GridOptimizer(n_rounds=10)],
                    ids=lambda solver: solver.__class__.__name__)
    def solver(self, request):
        return request.param

    @pytest.fixture
    def equation(self):
        """Evaluate: max(4 - (x - 2)^2), x = 2, y = 4."""
        equation = dict(x=3)
        equation['y'] = lambda: 4 - (equation['x'] - 2)**2
        # We use target for the expect y maximum
        return ManualTuneDefinition(
            get_x=lambda: equation['x'],
            set_x=lambda x: equation.__setitem__('x', x),
            get_y=lambda: equation['y'](),
            domain=(0, 4),
            target=4)


def test_gradient_descent_alpha():
    solver = GradientDescent()
    assert solver._alpha == hoomd.variant.Constant(0.1)
    assert solver.alpha == 0.1
    solver.alpha = hoomd.variant.Ramp(0.01, 0.001, 0, 20)
    current_alpha = solver.alpha
    assert current_alpha == 0.01
    for _ in range(20):
        solver.solve([])
        new_alpha = solver.alpha
        assert new_alpha < current_alpha
        current_alpha = new_alpha
    assert current_alpha == 0.001
