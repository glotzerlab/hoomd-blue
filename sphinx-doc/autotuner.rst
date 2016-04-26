Autotuner
=========

Overview
--------

HOOMD-blue uses run-time autotuning to optimize GPU performance. Every time you run a hoomd script, hoomd starts
autotuning values from a clean slate. Performance may vary during the first time steps of a simulation when the
autotuner is scanning through possible values. Once the autotuner completes the first scan, performance will stabilize
at optimized values. After approximately *period* steps, the autotuner will activate again and perform a quick scan
to update timing data. With continual updates, tuned parameters will adapt to simulation conditions - so as you
switch your simulation from NVT to NPT, compress the box, or change forces, the autotuner will keep everything
running at optimal performance.

Benchmarking hoomd
------------------

Care must be taken in performance benchmarks. The initial warm up time of the tuner is significant, and performance
measurements should only be taken after warm up. The total time needed for a scan may vary from system to system
depending on parameters. For example, the ``lj-liquid-bmark`` script requires 10,000 steps for the initial
tuning pass (2000 for subsequent updates). You can monitor the autotuner with the command line option
``--notice-level=4``. Each tuner will print a status message when it completes the warm up period. The ``nlist_binned``
tuner will most likely take the longest time to complete.

When obtaining profile traces, disable the autotuner after the warm up period so that it does not decide to re-tune
during the profile.

Controlling the autotuner
-------------------------

Default parameters should be sufficient for the autotuner to work well in almost any situation. Controllable parameters
are:

- ``period``: Approximate number of time steps before retuning occurs
- ``enabled``: Boolean to control whether the autotuner is enabled. If disabled after the warm up period, no retuning will
  occur, but it will still use the found optimal values. If disabled during the warm up period, a warning is issued
  and the system will use non-optimal values.

The defaults are ``period=100000``, and ``enabled=True``. Other parameters can be set by calling
:py:func:`hoomd.option.set_autotuner_params()`. This period is short enough to
pick up changes after just a few hundred thousand time steps, but long enough so that the performance loss of occasionally
running at nonoptimal parameters is small (most per time step calls can complete tuning in less than 200 time steps).
