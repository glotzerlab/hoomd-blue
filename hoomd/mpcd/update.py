# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD particle updaters

Updates properties of MPCD particles.

"""

import hoomd
from hoomd.md import _md

from . import _mpcd

class sort(hoomd.meta._metadata):
    R""" Sorts MPCD particles in memory to improve cache coherency.

    Args:
        system (:py:class:`hoomd.mpcd.data.system`): MPCD system to create sorter for
        period (int): Sort whenever the timestep is a multiple of *period*.
            .. versionadded:: 2.6

    Warning:
        Do not create :py:class:`hoomd.mpcd.update.sort` explicitly in your script.
        HOOMD creates a sorter by default.

    Every *period* time steps, particles are reordered in memory based on
    the cell list generated at the current timestep. Sorting can significantly improve
    performance of all other cell-based steps of the MPCD algorithm. The efficiency of
    the sort operation obviously depends on the number of particles, and so the *period*
    should be tuned to give the maximum performance.

    Note:
        The *period* should be no smaller than the MPCD collision period, or unnecessary
        cell list builds will occur.

    Essentially all MPCD systems benefit from sorting, and so a sorter is created by
    default with the MPCD system. To disable it or modify parameters, save the system
    and access the sorter through it::

        s = mpcd.init.read_snapshot(snap)
        # the sorter is only available after initialization
        s.sorter.set_period(period=5)
        s.sorter.disable()

    """

    def __init__(self, system, period=50):
        hoomd.util.print_status_line()

        # base class initialization
        hoomd.meta._metadata.__init__(self)

        # check for mpcd initialization
        if system.sorter is not None:
            hoomd.context.msg.error('mpcd.update: system already has a sorter created!\n')
            raise RuntimeError('MPCD sorter already created')

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _mpcd.Sorter
        else:
            cpp_class = _mpcd.SorterGPU
        self._cpp = cpp_class(system.data, hoomd.context.current.system.getCurrentTimeStep(), period)

        self.metadata_fields = ['period','enabled']
        self.period = period
        self.enabled = True

    def disable(self):
        hoomd.util.print_status_line()
        self.enabled = False

    def enable(self):
        hoomd.util.print_status_line()
        self.enabled = True

    def set_period(self, period):
        """ Change the sorting period.

        Args:
            period (int): New period to set.

        Examples::

            sorter.set_period(100)
            sorter.set_period(1)

        While the simulation is running, the action of each updater
        is executed every *period* time steps. Changing the period does
        not change the phase set when the analyzer was first created.

        """
        hoomd.util.print_status_line()

        self.period = period
        self._cpp.setPeriod(hoomd.context.current.system.getCurrentTimeStep(), self.period)

    def tune(self, start, stop, step, tsteps, quiet=False):
        """ Tune the sorting period.

        Args:
            start (int): Start of tuning interval to scan (inclusive).
            stop (int): End of tuning interval to scan (inclusive).
            step (int): Spacing between tuning points.
            tsteps (int): Number of timesteps to run at each tuning point.
            quiet (bool): Quiet the individual run calls.

        Returns:
            int: The optimal sorting period from the scanned range.

        The optimal sorting period for the MPCD particles is determined from
        a sequence of short runs. The sorting period is first set to *start*.
        The TPS value is determined for a run of length *tsteps*. This run is
        repeated 3 times, and the median TPS of the runs is saved. The sorting
        period is then incremented by *step*, and the process is repeated until
        *stop* is reached. The period giving the fastest TPS is determined, and
        the sorter period is updated to this value. The results of the scan
        are also reported as output, and the fastest sorting period is also
        returned.

        Note:
            A short warmup run is **required** before calling :py:meth:`tune()`
            in order to ensure the runtime autotuners have found optimal
            kernel launch parameters.

        Examples::

            # warmup run
            hoomd.run(5000)

            # tune sorting period
            sorter.tune(start=5, stop=50, step=5, tsteps=1000)

        """
        hoomd.util.print_status_line()

        # hide calls to set_period, etc.
        hoomd.util.quiet_status()

        # scan through range of sorting periods and log TPS
        periods = range(start, stop+1, step)
        tps = []
        for p in periods:
            cur_tps = []
            self.set_period(period=p)
            for i in range(0,3):
                hoomd.run(tsteps, quiet=quiet)
                cur_tps.append(hoomd.context.current.system.getLastTPS())

            # save the median tps
            cur_tps.sort()
            tps.append(cur_tps[1])

        # determine fastest period and set it on the sorter
        fastest = tps.index(max(tps))
        opt_period = periods[fastest]
        self.set_period(period=opt_period)

        # tuning is done, restore status lines
        hoomd.util.unquiet_status()

        # output results
        hoomd.context.msg.notice(2, '--- sort.tune() statistics\n')
        hoomd.context.msg.notice(2, 'Optimal period = {0}\n'.format(opt_period))
        hoomd.context.msg.notice(2, '        period = ' + str(periods) + '\n')
        hoomd.context.msg.notice(2, '          TPS  = ' + str(tps) + '\n')

        return opt_period
