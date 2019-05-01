# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Benchmark utilities

Commands that help in benchmarking HOOMD-blue performance.
"""

import hoomd

def series(warmup=100000, repeat=20, steps=10000, limit_hours=None):
    R""" Perform a series of benchmark runs.

    Args:
        warmup (int): Number of time steps to :py:meth:`hoomd.run()` to warm up the benchmark
        repeat (int): Number of times to repeat the benchmark *steps*.
        steps (int): Number of time steps to :py:meth:`hoomd.run()` at each benchmark point.
        limit_hours (float): Limit each individual :py:meth:`hoomd.run()` length to this time.

    :py:meth:`series()` executes *warmup* time steps. After that, it
    calls ``run(steps)``, *repeat* times and returns a list containing the average TPS for each of those runs.
    """
    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot tune r_buff before initialization\n");

    tps_list = [];

    if warmup > 0:
        hoomd.run(warmup);

    for i in range(0,repeat):
        hoomd.run(steps, limit_hours=limit_hours);
        tps_list.append(hoomd.context.current.system.getLastTPS());

    return tps_list;
