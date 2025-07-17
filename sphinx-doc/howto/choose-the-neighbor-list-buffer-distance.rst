.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to choose the neighbor list buffer distance
===============================================

Set the neighbor list buffer distance (`hoomd.md.nlist.NeighborList.buffer`) to maximize the
performance of your simulation. The neighbor list recomputes itself more often when ``buffer`` is
small, and the pair force computation takes more time when ``buffer`` is large. There
is an optimal value between the extremes that strongly depends on system size, density, the hardware
device, the pair potential, step size, and more. Test your specific model, change ``buffer`` and
measure the performance to find the optimal. For example:

.. literalinclude:: choose-the-neighbor-list-buffer-distance.py
    :language: python

Example output::

    buffer=0: TPS=212, num_builds=1001
    buffer=0.05: TPS=584, num_builds=197
    buffer=0.1: TPS=839, num_builds=102
    buffer=0.2: TPS=954, num_builds=53
    buffer=0.3: TPS=880, num_builds=37

.. important::

    Ensure that you run sufficient steps to sample many neighbor list builds to properly sample
    the amortized time spent per build.

.. tip::

    Measure the optimal value of ``buffer`` in testing, then apply that fixed value in production
    runs to avoid wasting time at non-optimal values.

.. seealso::

    `hoomd.md.tune.NeighborListBuffer` can automate this process.
