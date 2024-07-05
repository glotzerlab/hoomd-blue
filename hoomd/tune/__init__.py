# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Tuners.

`Tuner` operations make changes to the parameters of other operations (or the
simulation state) that adjust the performance of the simulation without changing
the correctness of the outcome. Every new `hoomd.Simulation` object includes a
`ParticleSorter` in its operations by default. `ParticleSorter` rearranges the
order of particles in memory to improve cache-coherency.

This package also defines the `CustomTuner` class and a number of helper
classes. Use these to implement custom tuner operations in Python code.

.. rubric:: Solver

Most tuners explicitly involve solving some sort of mathematical problem (e.g.
root-finding or optimization). HOOMD provides infrastructure for solving these
problems as they appear in our provided `hoomd.operation.Tuner` subclasses. All
tuners that involve iteratively solving a problem compose a `SolverStep`
subclass instance. The `SolverStep` class implements the boilerplate to do
iterative solving given a simulation where calls to the "function" being solves
means running potentially 1,000's of steps.

Every solver regardless of type has a ``solve`` method which accepts a list of
tunable quantities. The method returns a Boolean indicating whether all
quantities are considered tuned or not. Tuners indicate they are tuned when two
successive calls to `SolverStep.solve` return ``True`` unless otherwise
documented.

Custom solvers can be created from inheriting from the base class of one of the
problem types (`RootSolver` and `Optimizer`) or `SolverStep` if solving a
different problem type. All that is required is to implement the
`SolverStep.solve_one` method, and the solver can be used by any HOOMD tuner
that expects a solver.

.. rubric:: Custom Tuners

Through using `SolverStep` subclasses and `ManualTuneDefinition` most tuning
problems should be solvable for a `CustomTuner`. To create a tuner define all
`ManualTuneDefinition` interfaces for each tunable and plug into a solver in a
`CustomTuner`.
"""

from hoomd.tune.sorter import ParticleSorter
from hoomd.tune.balance import LoadBalancer
from hoomd.tune.custom_tuner import CustomTuner, _InternalCustomTuner
from hoomd.tune.attr_tuner import ManualTuneDefinition
from hoomd.tune.solve import (GridOptimizer, GradientDescent, Optimizer,
                              RootSolver, ScaleSolver, SecantSolver, SolverStep)
