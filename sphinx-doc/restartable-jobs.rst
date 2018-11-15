.. _restartable-jobs:

Restartable jobs
================

Overview
--------

The ideal restartable job is a single job script that can be resubmitted over and over again to the job queue system.
Each time the job starts, it picks up where it left off the last time and continues running until it is done.
You can put all the logic necessary to do this in the hoomd python script itself, keeping the submission script simple::

    # job.sh
    mpirun hoomd run.py

With a properly configured python script, ``qsub job.sh`` is all that is necessary to submit the first run,
continue a previous job that exited cleanly, and continue one that was prematurely killed.

Elements of a restartable script
--------------------------------

A restartable needs to:

 - Choose between an initial condition and the restart file when initializing.
 - Write a restart file periodically.
 - Set a phase for all analysis, dump, and update commands.
 - Use :py:func:`hoomd.run_upto()` to skip over time steps that were run in previous job submissions.
 - Use only commands that are restart capable.
 - Optionally ensure that jobs cleanly exit before the job walltime limit.

Choose the appropriate initialization file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's assume that the initial condition for the simulation is in ``init.gsd``, and ``restart.gsd`` is saved periodically
as the job runs. A single :py:func:`hoomd.init.read_gsd()` command will load the restart file if it exists, otherwise it will load
the initial file. It is easiest to think about dump files, temperature ramps, etc... if ``init.gsd`` is at time step 0::

    init.read_gsd(filename='init.gsd', restart='restart.gsd')

If you generate your initial configuration in python, you will need to add some logic to read ``restart.gsd`` if it
exists or generate if not. This logic is left as an exercise to the reader.

Write restart files
^^^^^^^^^^^^^^^^^^^

You cannot predict when a hardware failure will cause your job to fail, so you need to save restart files at regular
intervals as your run progresses. You will also need periodic restart files at a fast rate if you don't manage wall
time to ensure clean job exits.

First, you need to select a restart period. The compute center you run on may offer a tool to help you determine
an optimal restart period in minutes. A good starting point is to write a restart file every hour. Based on performance
benchmarks, select a restart period in time steps::

    dump.gsd(filename="restart.gsd", group=group.all(), truncate=True, period=10000, phase=0)

Use the phase option
^^^^^^^^^^^^^^^^^^^^

Set a a ``phase >= 0`` for all analysis routines, file dumps, and updaters you use with period > 1 (the default is 0).
With ``phase >= 0``, these routines will continue to run in a restarted job on the correct timesteps as if the job had
not been restarted.

Do not use, ``phase=-1``, as then these routines will start running immediately when a restart job begins::

    dump.dcd(filename="trajectory.dcd", period=1e6, phase=0)
    analyze.log(filename='temperature.log', quantities=['temperature'], period=5000, phase=0)
    zeroer= update.zero_momentum(period=1e6, phase=0)

Use run_upto
^^^^^^^^^^^^

:py:func:`hoomd.run_upto` runs the simulation up to timestep ``n``. Use this in restartable jobs to allow them to run a
given number of steps, independent of the number of submissions needed to reach that::

    run_upto(100e6)

Use restart capable commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most commands in hoomd that output to files are capable of appending to the end of a file so that restarted jobs
continue adding data to the file as if the job had never been restarted.

However, not all features in hoomd are capable of restarting. Some are not even capable of appending to files. See the
documentation for each individual command you use to tell whether it is compatible with restartable jobs.
For those that are restart capable, do not set `overwrite=True`, or each time the job restarts it will erase the file
and start writing a new one.

Some analysis routines in HOOMD-blue store internal state and may require a period that is commensurate with the
restart period. See the documentation on the individual command you use to see if this is the case.

Cleanly exit before the walltime limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Job queues will kill your job when it reaches the walltime limit. HOOMD can stop your run before that happens and
give your job time to exit cleanly. Set the environment variable ``HOOMD_WALLTIME_STOP`` to enable this.
Any :py:func:`hoomd.run()` or :py:func:`hoomd.run_upto()` command will exit before the specified time is reached.
HOOMD monitors run performance and tries to ensure that it will end *before* ``HOOMD_WALLTIME_STOP``.
Set the variable to a unix epoch time. For example in a job script that should run 12 hours, set ``HOOMD_WALLTIME_STOP``
to 12 hours from now, minus 10 minutes to allow for job cleanup::

    # job.sh
    export HOOMD_WALLTIME_STOP=$((`date +%s` + 12 * 3600 - 10 * 60))
    mpirun hoomd run.py

When using ``HOOMD_WALLTIME_STOP``, :py:func:`hoomd.run()` will throw the exception ``WalltimeLimitReached`` when it exits due to the walltime
limit. Catch this exception so that your job can exit cleanly. Also, make sure to write out a final restart file
at the end of your job so you have the final system state to continue from. Set the ``limit_multiple`` for the run to
the restart period so that any analyzers that must run commensurate with the restart file have a chance to run. If you
don't use any such commands, you can omit ``limit_multiple`` and the run will be free to end on any time step::

    gsd_restart = dump.gsd(filename="restart.gsd", group=group.all(), truncate=True, period=10000, phase=0)

    try:
        run_upto(1e6, limit_multiple=10000)

        # Perform additional actions here that should only be done after the job has completed all time steps.
    except WalltimeLimitReached:
        # Perform actions here that need to be done each time you run into the wall clock limit, or just pass
        pass

    gsd_restart.write_restart()
    # Perform additional job cleanup actions here. These will be executed each time the job ends due to reaching the
    # walltime limit AND when the job completes all of its time steps.

Examples
--------

Simple example
^^^^^^^^^^^^^^

Here is a simple example that puts all of these elements together::

    # job.sh
    export HOOMD_WALLTIME_STOP=$((`date +%s` + 12 * 3600 - 10 * 60))
    mpirun hoomd run.py

.. code::

    # run.py
    from hoomd import *
    from hoomd import md
    context.initialize()

    init.read_gsd(filename='init.gsd', restart='restart.gsd')

    lj = md.pair.lj(r_cut=2.5)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

    md.integrate.mode_standard(dt=0.005)
    md.integrate.nvt(group=group.all(), T=1.2, tau=0.5)

    gsd_restart = dump.gsd(filename="restart.gsd", group=group.all(), truncate=True, period=10000, phase=0)
    dump.dcd(filename="trajectory.dcd", period=1e5, phase=0)
    analyze.log(filename='temperature.log', quantities=['temperature'], period=5000, phase=0)

    try:
        run_upto(1e6, limit_multiple=10000)
    except WalltimeLimitReached:
        pass

    gsd_restart.write_restart()

Temperature ramp
^^^^^^^^^^^^^^^^

Runs often have temperature ramps. These are trivial to make restartable using a variant. Just be sure to set
the ``zero=0`` option so that the ramp starts at timestep 0 and does not begin at the top every time the job is submitted.
The only change needed from the previous simple example is to use the variant in ``integrate.nvt()``::


    T_variant = variant.linear_interp(points = [(0, 2.0), (2e5, 0.5)], zero=0)
    integrate.nvt(group=group.all(), T=T_variant, tau=0.5)

Multiple stage jobs
^^^^^^^^^^^^^^^^^^^

Not all ramps or staged job protocols can be expressed as variants. However, it is easy to implement multi-stage jobs
using run_upto and ``HOOMD_WALLTIME_STOP``. Here is an example of a more complex job that involves multiple stages::

    # run.py
    from hoomd import *
    from hoomd import md
    context.initialize()

    init.read_gsd(filename='init.gsd', restart='restart.gsd')

    lj = md.pair.lj(r_cut=2.5)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

    md.integrate.mode_standard(dt=0.005)

    gsd_restart = dump.gsd(filename="restart.gsd", group=group.all(), truncate=True, period=10000, phase=0)

    try:
        # randomize at high temperature
        nvt = md.integrate.nvt(group=group.all(), T=5.0, tau=0.5)
        run_upto(1e6, limit_multiple=10000)

        # equilibrate
        nvt.set_params(T=1.0)
        run_upto(2e6, limit_multiple=10000)

        # switch to nve and start saving data for the production run
        nvt.disable();
        md.integrate.nve(group=group.all())
        dump.dcd(filename="trajectory.dcd", period=1e5, phase=0)
        analyze.log(filename='temperature.log', quantities=['temperature'], period=5000, phase=0)

        run_upto(12e6);

    except WalltimeLimitReached:
        pass

    gsd_restart.write_restart()

And here is another example that changes interaction parameters::

    try:
        for i in range(1,11):
            lj.pair_coeff.set('A', 'A', epsilon=0.1*i)
            run_upto(1e6*i);
    except WalltimeLimitReached:
        pass

Multiple hoomd invocations
^^^^^^^^^^^^^^^^^^^^^^^^^^

``HOOMD_WALLTIME_STOP`` is a global variable set at the start of a job script. So you can launch hoomd scripts multiple times
from within a job script and any of those individual runs will exit cleanly when it reaches the walltime. You need
to take care that you don't start any new scripts once the first exits due to a walltime limit.
The BASH script logic necessary to implement this behavior is workflow dependent and left as an exercise to
the reader.
