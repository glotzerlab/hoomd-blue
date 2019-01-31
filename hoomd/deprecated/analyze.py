# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" Deprecated analyzers.
"""

from hoomd.analyze import _analyzer;
from hoomd.deprecated import _deprecated;
import hoomd;

class msd(_analyzer):
    R""" Mean-squared displacement.

    Args:
        filename (str): File to write the data to.
        groups (list): List of groups to calculate the MSDs of.
        period (int): Quantities are logged every *period* time steps.
        header_prefix (str): (optional) Specify a string to print before the header.
        r0_file (str): hoomd_xml file specifying the positions (and images) to use for :math:`\vec{r}_0`.
        overwrite (bool): set to True to overwrite the file *filename* if it exists.
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.

    .. deprecated:: 2.0
       analyze.msd will be replaced by a more general system capable of window averaging in a future release.

    :py:class:`msd` can be given any number of groups of particles. Every *period* time steps, it calculates the mean squared
    displacement of each group (referenced to the particle positions at the time step the command is issued at) and prints
    the calculated values out to a file.

    The mean squared displacement (MSD) for each group is calculated as:

    .. math::
        \langle |\vec{r} - \vec{r}_0|^2 \rangle

    and values are correspondingly written in units of distance squared.

    The file format is the same convenient delimited format used by :py:class`hoomd.analyze.log`.

    :py:class:`msd` is capable of appending to an existing msd file (the default setting) for use in restarting in long jobs.
    To generate a correct msd that does not reset to 0 at the start of each run, save the initial state of the system
    in a hoomd_xml file, including position and image data at a minimum. In the continuation job, specify this file
    in the *r0_file* argument to analyze.msd.

    Examples::

        msd = analyze.msd(filename='msd.log', groups=[group1, group2],
                          period=100)

        analyze.msd(groups=[group1, group2, group3], period=1000,
                    filename='msd.log', header_prefix='#')

        analyze.msd(filename='msd.log', groups=[group1], period=10,
                    header_prefix='Log of group1 msd, run 5\n')


    A group variable (*groupN* above) can be created by any number of group creation functions.
    See group for a list.

    By default, columns in the file are separated by tabs, suitable for importing as a
    tab-delimited spreadsheet. The delimiter can be changed to any string using :py:meth:`set_params()`.

    The *header_prefix* can be used in a number of ways. It specifies a simple string that
    will be printed before the header line of the output file. One handy way to use this
    is to specify header_prefix='#' so that ``gnuplot`` will ignore the header line
    automatically. Another use-case would be to specify a descriptive line containing
    details of the current run. Examples of each of these cases are given above.

    If *r0_file* is left at the default of None, then the current state of the system at the execution of the
    analyze.msd command is used to initialize :math:`\vec{r}_0`.

    """

    def __init__(self, filename, groups, period, header_prefix='', r0_file=None, overwrite=False, phase=0):
        hoomd.util.print_status_line();

        # initialize base class
        _analyzer.__init__(self);

        # create the c++ mirror class
        self.cpp_analyzer = _deprecated.MSDAnalyzer(hoomd.context.current.system_definition, filename, header_prefix, overwrite);
        self.setupAnalyzer(period, phase);

        # it is an error to specify no groups
        if len(groups) == 0:
            hoomd.context.msg.error('At least one group must be specified to analyze.msd\n');
            raise RuntimeError('Error creating analyzer');

        # set the group columns
        for cur_group in groups:
            self.cpp_analyzer.addColumn(cur_group.cpp_group, cur_group.name);

        if r0_file is not None:
            self.cpp_analyzer.setR0(r0_file);

    def set_params(self, delimiter=None):
        R""" Change the parameters of the msd analysis

        Args:
            delimiter (str): New delimiter between columns in the output file (if specified).

        Examples::

            msd.set_params(delimiter=',');
        """
        hoomd.util.print_status_line();

        if delimiter:
            self.cpp_analyzer.setDelimiter(delimiter);
