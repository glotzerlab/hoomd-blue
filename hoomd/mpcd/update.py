# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD particle updaters

Updates properties of MPCD particles.

"""

import hoomd
from hoomd.md import _md

from . import _mpcd

class sort(hoomd.update._updater):
    R""" Sorts MPCD particles in memory to improve cache coherency.

    Args:
        system (:py:class:`hoomd.mpcd.data.system`): MPCD system to create sorter for

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

    def __init__(self, system):
        hoomd.util.print_status_line()

        # base class initialization
        hoomd.update._updater.__init__(self)

        # check for mpcd initialization
        if system.sorter is not None:
            hoomd.context.msg.error('mpcd.update: system already has a sorter created!\n')
            raise RuntimeError('MPCD sorter already created')

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cpp_class = _mpcd.Sorter
        else:
            cpp_class = _mpcd.SorterGPU
        self.cpp_updater = cpp_class(system.data)

        self.metadata_fields = ['period']
        self.period = 50

        self.setupUpdater(self.period)

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

        # call base updater's set period method
        hoomd.util.quiet_status()
        hoomd.update._updater.set_period(self, period)
        hoomd.util.unquiet_status()
        # and save the period into ourselves as metadata
        self.period = period
