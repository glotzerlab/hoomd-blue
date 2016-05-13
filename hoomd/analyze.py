# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Commands that analyze the system and provide some output.

An analyzer examines the system state in some way every *period* time steps and generates
some form of output based on the analysis. Check the documentation for individual analyzers
to see what they do.
"""

from hoomd import _hoomd;
import hoomd;
import sys;

## \page variable_period_docs Variable period specification
#
# TODO: put this in its own page in sphinx
#
# If, for any reason, a constant period for a command is not to your liking, you can make it any
# function you please! Just specify a function taking a single argument to the period parameter.
# Any analyze, update, or dump command in hoomd can be given such a variable period.
# dump.xml is used as an example here, but the same works with \b any update, dump,
# or analyze command
#
# For example, lets say we want to dump xml files at time steps 1, 10, 100, 1000, ...
# The following command will do the job.
#
# \code
# dump.xml(filename="dump", period = lambda n: 10**n)
# \endcode
#
# It is that simple. Any mathematical expression that can be represented in python can be used
# in place of the 10**n.
#
# <b>More examples:</b>
# \code
# dump.xml(filename="dump", period = lambda n: n**2)
# dump.xml(filename="dump", period = lambda n: 2**n)
# dump.xml(filename="dump", period = lambda n: 1005 + 0.5 * 10**n)
# \endcode
#
# The only requirement is that the object passed into period is callable, accepts one argument, and returns
# a floating point number or integer. The function also had better be monotonically increasing or the output
# might not make any sense.
#
# <b>How does it work, exactly?</b>
# - First, the current time step of the simulation is saved when the analyzer is created
# - \a n is also set to 1 when the analyzer is created
# - Every time the analyzer performs it's output, it evaluates the given function at the current value of \a n
#   and records that as the next time to perform the analysis. \a n is then incremented by 1
#
# Here is a final example of how variable periods behave in simulations where analyzers are not created on time step 0.
# The following
# \code
# ... initialize ...
# run(4000)
# dump.xml(filename="dump", period = lambda n: 2**n)
# run(513)
# \endcode
# will result in dump files at time steps 4000, 4002, 4004, 4008, 4016, 4032, 4064, 4128, 4256, and 4512.
#
# In other words, the function specified for the period starts counting at the time step <b>when the analyzer is created</b>.
# Consequently, any analyze, dump, or update command given a variable period becomes ill-defined if it is disabled and then re-enabled.
# If this is done, it will then re-enable with a constant period of 1000 as a default case.
#

## \internal
# \brief Base class for analyzers
#
# An analyzer in hoomd_script reflects an Analyzer in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ analyzer itself is tracked and added to the
# System 2) methods are provided for disabling the analyzer and changing the
# period which the system calls it
class _analyzer(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the analyzer
    #
    # Initializes the cpp_analyzer to None.
    # Assigns a name to the analyzer in analyzer_name;
    def __init__(self):
        # check if initialization has occurred
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create analyzer before initialization\n");
            raise RuntimeError('Error creating analyzer');

        self.cpp_analyzer = None;

        # increment the id counter
        id = _analyzer.cur_id;
        _analyzer.cur_id += 1;

        self.analyzer_name = "analyzer%d" % (id);
        self.enabled = True;

        # Store a reference in global simulation variables
        hoomd.context.current.analyzers.append(self)

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \internal
    # \brief Helper function to setup analyzer period
    #
    # \param period An integer or callable function period
    # \param phase Phase parameter
    #
    # If an integer is specified, then that is set as the period for the analyzer.
    # If a callable is passed in as a period, then a default period of 1000 is set
    # to the integer period and the variable period is enabled
    #
    def setupAnalyzer(self, period, phase=-1):
        self.phase = phase;

        if type(period) == type(1.0):
            hoomd.context.current.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, int(period), phase);
        elif type(period) == type(1):
            hoomd.context.current.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, period, phase);
        elif type(period) == type(lambda n: n*2):
            hoomd.context.current.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, 1000, -1);
            hoomd.context.current.system.setAnalyzerPeriodVariable(self.analyzer_name, period);
        else:
            hoomd.context.msg.error("I don't know what to do with a period of type " + str(type(period)) + " expecting an int or a function\n");
            raise RuntimeError('Error creating analyzer');

    ## \var enabled
    # \internal
    # \brief True if the analyzer is enabled

    ## \var cpp_analyzer
    # \internal
    # \brief Stores the C++ side Analyzer managed by this class

    ## \var analyzer_name
    # \internal
    # \brief The Analyzer's name as it is assigned to the System

    ## \var prev_period
    # \internal
    # \brief Saved period retrieved when an analyzer is disabled: used to set the period when re-enabled

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_analyzer is None:
            hoomd.context.msg.error('Bug in hoomd_script: cpp_analyzer not set, please report\n');
            raise RuntimeError();

    def disable(self):
        R""" Disable the analyzer.

        Examples::

            my_analyzer.disable()


        Executing the disable command will remove the analyzer from the system.
        Any :py:func:`hoomd.run()` command executed after disabling an analyzer will not use that
        analyzer during the simulation. A disabled analyzer can be re-enabled
        with :py:meth:`enable()`.
        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if not self.enabled:
            hoomd.context.msg.warning("Ignoring command to disable an analyzer that is already disabled");
            return;

        self.prev_period = hoomd.context.current.system.getAnalyzerPeriod(self.analyzer_name);
        hoomd.context.current.system.removeAnalyzer(self.analyzer_name);
        self.enabled = False;

    def enable(self):
        R""" Enables the analyzer

        Examples::

            my_analyzer.enable()

        See :py:meth:`disable()`.
        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            hoomd.context.msg.warning("Ignoring command to enable an analyzer that is already enabled");
            return;

        hoomd.context.current.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, self.prev_period, self.phase);
        self.enabled = True;

    def set_period(self, period):
        R""" Changes the period between analyzer executions

        Args:
            period (int): New period to set (in time steps)

        Examples::

            analyzer.set_period(100)
            analyzer.set_period(1)


        While the simulation is running (:py:func:`hoomd.run()`, the action of each analyzer
        is executed every *period* time steps. Changing the period does not change the phase set when the analyzer
        was first created.
        """
        hoomd.util.print_status_line();
        self.period = period;

        if type(period) == type(1):
            if self.enabled:
                hoomd.context.current.system.setAnalyzerPeriod(self.analyzer_name, period, self.phase);
            else:
                self.prev_period = period;
        elif type(period) == type(lambda n: n*2):
            hoomd.context.msg.warning("A period cannot be changed to a variable one");
        else:
            hoomd.context.msg.warning("I don't know what to do with a period of type " + str(type(period)) + " expecting an int or a function");

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled
        return data

# set default counter
_analyzer.cur_id = 0;

class imd(_analyzer):
    R""" Send simulation snapshots to VMD in real-time.

    Args:
        port (int): TCP/IP port to listen on.
        period (int): Number of time steps to run before checking for new IMD messages.
        rate (int): Number of periods between coordinate data transmissions.
        pause (bool): Set to *True* to pause the simulation at the first time step until an imd connection is made.
        force (:py:class:`hoomd.md.force.constant`): A force that apply forces received from VMD.
        force_scale (float): Factor by which to scale all forces received from VMD.
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where `(step + phase) % period == 0`.

    :py:class:`hoomd.analyze.imd` listens on a specified TCP/IP port for connections from VMD.
    Once that connection is established, it begins transmitting simulation snapshots
    to VMD every *rate* time steps.

    To connect to a simulation running on the local host, issue the command::

        imd connect localhost 54321

    in the VMD command window (where 54321 is replaced with the port number you specify for
    :py:class:`hoomd.analyze.imd`.

    Note:
        If a period larger than 1 is set, the actual rate at which time steps are transmitted is ``rate * period``.

    Examples::

        analyze.imd(port=54321, rate=100)
        analyze.imd(port=54321, rate=100, pause=True)
        imd = analyze.imd(port=12345, rate=1000)
    """
    def __init__(self, port, period=1, rate=1, pause=False, force=None, force_scale=0.1, phase=-1):
        hoomd.util.print_status_line();

        # initialize base class
        _analyzer.__init__(self);

        # get the cpp force
        if force is not None:
            cpp_force = force.cpp_force;
        else:
            cpp_force = None;

        # create the c++ mirror class
        self.cpp_analyzer = _hoomd.IMDInterface(hoomd.context.current.system_definition, port, pause, rate, cpp_force);
        self.setupAnalyzer(period, phase);


class log(_analyzer):
    R""" Log a number of calculated quantities to a file.

    Args:
        filename (str): File to write the log to, or *None* for no file output.
        quantities (list): List of quantities to log.
        period (int): Quantities are logged every *period* time steps.
        header_prefix (str):  Specify a string to print before the header.
        overwrite (bool): When False (the default) an existing log will be appended to. When True, an existing log file will be overwritten instead.
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.

    :py:class:`hoomd.analyze.log` reads a variety of calculated values, like energy and temperature, from
    specified forces, integrators, and updaters. It writes a single line to the specified
    output file every *period* time steps. The resulting file is suitable for direct import
    into a spreadsheet, MATLAB, or other software that can handle simple delimited files.

    Quantities that can be logged at any time:
    - **volume** - Volume of the simulation box (in volume units)
    - **N** - Particle nubmer (dimensionless)
    - **lx** - Box length in x direction (in length units)
    - **ly** - Box length in y direction (in length units)
    - **lz** - Box length in z direction (in length units)
    - **xy** - Box tilt factor in xy plane (dimensionless)
    - **xz** - Box tilt factor in xz plane (dimensionless)
    - **yz** - Box tilt factor in yz plane (dimensionless)
    - **momentum** - Magnitude of the average momentum of all particles (in momentum units)
    - **time** - Wall-clock running time from the start of the log (in seconds)

    Thermodynamic properties:
    - The following quantities are always available and computed over all particles in the system (see compute.thermo for detailed definitions):

      - **num_particles**
      - **ndof**
      - **translational_ndof**
      - **rotational_ndof**
      - **potential_energy** (in energy units)
      - **kinetic_energy** (in energy units)
      - **translational_kinetic_energy** (in energy units)
      - **rotational_kinetic_energy** (in energy units)
      - **temperature** (in thermal energy units)
      - **pressure** (in pressure units)
      - **pressure_xx**, **pressure_xy**, **pressure_xz**, **pressure_yy**, **pressure_yz**, **pressure_zz** (in pressure units)

    - The above quantities, tagged with a <i>_groupname</i> suffix are automatically available for any group passed to
      an integrate command
    - Specify a compute.thermo directly to enable additional quantities for user-specified groups.

    The following quantities are only available if the command is parentheses has been specified and is active
    for logging:

    - Pair potentials

      - **pair_dpd_energy** (pair.dpd) - Total DPD conservative potential energy (in energy units)
      - **pair_dpdlj_energy** (pair.dpdlj) - Total DPDLJ conservative potential energy (in energy units)
      - **pair_eam_energy** (pair.eam) - Total EAM potential energy (in energy units)
      - **pair_ewald_energy** (pair.ewald) - Short ranged part of the electrostatic energy (in energy units)
      - **pair_gauss_energy** (pair.gauss) - Total Gaussian potential energy (in energy units)
      - **pair_lj_energy** (pair.lj) - Total Lennard-Jones potential energy (in energy units)
      - **pair_morse_energy** (pair.yukawa) - Total Morse potential energy (in energy units)
      - **pair_table_energy** (pair.table) - Total potential energy from Tabulated potentials (in energy units)
      - **pair_slj_energy** (pair.slj) - Total Shifted Lennard-Jones potential energy (in energy units)
      - **pair_yukawa_energy** (pair.yukawa) - Total Yukawa potential energy (in energy units)
      - **pair_force_shifted_lj_energy** (pair.force_shifted_lj) - Total Force-shifted Lennard-Jones potential energy (in energy units)
      - **pppm_energy** (charge.pppm) -  Long ranged part of the electrostatic energy (in energy units)

    - Bond potentials

      - **bond_fene_energy** (bond.fene) - Total fene bond potential energy (in energy units)
      - **bond_harmonic_energy** (bond.harmonic) - Total harmonic bond potential energy (in energy units)
      - **bond_table_energy** (bond.table) - Total table bond potential energy (in energy units)

    - Angle potentials

      - **angle_harmonic_energy** (angle.harmonic) - Total harmonic angle potential energy (in energy units)

    - Dihedral potentials

      - **dihedral_harmonic_energy** (dihedral.harmonic) - Total harmonic dihedral potential energy (in energy units)

    - External potentials

      - **external_periodic_energy** (external.periodic) - Total periodic potential energy (in energy units)
      - **external_e_field_energy** (external.e_field) - Total e_field potential energy (in energy units)

    - Wall potentials

      - **external_wall_lj_energy** (wall.lj) - Total Lennard-Jones wall energy (in energy units)
      - **external_wall_gauss_energy** (wall.gauss) - Total Gauss wall energy (in energy units)
      - **external_wall_slj_energy** (wall.slj) - Total Shifted Lennard-Jones wall energy (in energy units)
      - **external_wall_yukawa_energy** (wall.yukawa) - Total Yukawa wall energy (in energy units)
      - **external_wall_mie_energy** (wall.mie) - Total Mie wall energy (in energy units)

    - Integrators

      - **langevin_reservoir_energy_groupname** (integrate.bdnvt) - Energy reservoir for the Langevin integrator (in energy units)
      - **nvt_reservoir_energy_groupname** (integrate.nvt) - Energy reservoir for the NVT thermostat (in energy units)
      - **nvt_mtk_reservoir_energy_groupname** (integrate.nvt) - Energy reservoir for the NVT MTK thermostat (in energy units)
      - **npt_thermostat_energy** (integrate.npt) - Energy of the NPT thermostat
      - **npt_barostat_energy** (integrate.npt & integrate.nph) - Energy of the NPT (or NPH) barostat

    Additionally, all pair and bond poetentials can be provided user-defined names that are appended as suffixes to the
    logged quantitiy (e.g. with ``pair.lj(r_cut=2.5, name="alpha")``, the logged quantity would be pair_lj_energy_alpha):

    By specifying a force, disabling it with the *log=True* option, and then logging it, different energy terms can
    be computed while only a subset of them actually drive the simulation. Common use-cases of this capability
    include separating out pair energy of given types (shown below) and free energy calculations. Be aware that the
    globally chosen *r_cut* value is the largest of all active pair potentials and those with *log=True*, so you will
    observe performance degradation if you *disable(log=True)* a potential with a large *r_cut*.

    File output from analyze.log is optional. Specify *None* for the file name and no file will be output.
    Use this with the :py:meth:`query()` method to query the values of properties without the overhead of writing them
    to disk.

    You can register custom python callback functions to provide logged quantities with :py:meth:`register_callback()`.

    Examples::

        lj1 = pair.lj(r_cut=3.0, name="lj1")
        lj1.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj1.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0)
        lj1.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0)

        lj2 = pair.lj(r_cut=3.0, name="lj2")
        lj2.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        lj2.pair_coeff.set('A', 'B', epsilon=0.0, sigma=1.0)
        lj2.pair_coeff.set('B', 'B', epsilon=0.0, sigma=1.0)
        lj2.disable(log=True)

        analyze.log(filename='mylog.log', quantities=['pair_lj_energy_lj1', 'pair_lj_energy_lj2'],
                    period=100, header_prefix='#')


        logger = analyze.log(filename='mylog.log', period=100,
                             quantities=['pair_lj_energy'])

        analyze.log(quantities=['pair_lj_energy', 'bond_harmonic_energy',
                    'kinetic_energy'], period=1000, filename='full.log')

        analyze.log(filename='mylog.log', quantities=['pair_lj_energy'],
                    period=100, header_prefix='#')

        analyze.log(filename='mylog.log', quantities=['bond_harmonic_energy'],
                    period=10, header_prefix='Log of harmonic energy, run 5\\n')
        logger = analyze.log(filename='mylog.log', period=100,
                             quantities=['pair_lj_energy'], overwrite=True)

        log = analyze.log(filename=None, quantities=['potential_energy'], period=1)
        U = log.query('potential_energy')

    TODO: reference units concept page.
    TODO: cross reference to appropriate classes.

    By default, columns in the log file are separated by tabs, suitable for importing as a
    tab-delimited spreadsheet. The delimiter can be changed to any string using :py:meth:`set_params()`

    The *header_prefix* can be used in a number of ways. It specifies a simple string that
    will be printed before the header line of the output file. One handy way to use this
    is to specify header_prefix='#' so that ``gnuplot`` will ignore the header line
    automatically. Another use-case would be to specify a descriptive line containing
    details of the current run. Examples of each of these cases are given above.

    Warning:
        When an existing log is appended to, the header is not printed. For the log to
        remain consistent with the header already in the file, you must specify the same quantities
        to log and in the same order for all runs of hoomd that append to the same log.
    """

    def __init__(self, filename, quantities, period, header_prefix='', overwrite=False, phase=-1):
        hoomd.util.print_status_line();

        # initialize base class
        _analyzer.__init__(self);

        if filename is None or filename == "":
            filename = "";
            period = 1;

        # create the c++ mirror class
        self.cpp_analyzer = _hoomd.Logger(hoomd.context.current.system_definition, filename, header_prefix, overwrite);
        self.setupAnalyzer(period, phase);

        # set the logged quantities
        quantity_list = _hoomd.std_vector_string();
        for item in quantities:
            quantity_list.append(str(item));
        self.cpp_analyzer.setLoggedQuantities(quantity_list);

        # add the logger to the list of loggers
        hoomd.context.current.loggers.append(self);

        # store metadata
        self.metadata_fields = ['filename','period']
        self.filename = filename
        self.period = period

    def set_params(self, quantities=None, delimiter=None):
        R""" Change the parameters of the log.

        Args:
            quantities (list): New list of quantities to log (if specified)
            delimiter (str): New delimiter between columns in the output file (if specified)

        Examples::

            logger.set_params(quantities=['bond_harmonic_energy'])
            logger.set_params(delimiter=',');
            logger.set_params(quantities=['bond_harmonic_energy'], delimiter=',');
        """

        hoomd.util.print_status_line();

        if quantities is not None:
            # set the logged quantities
            quantity_list = _hoomd.std_vector_string();
            for item in quantities:
                quantity_list.append(str(item));
            self.cpp_analyzer.setLoggedQuantities(quantity_list);

        if delimiter:
            self.cpp_analyzer.setDelimiter(delimiter);

    def query(self, quantity):
        R""" Get the current value of a logged quantity.

        Args:
            quantity (str): Name of the quantity to return.

        :py:meth:`query()` works in two different ways depending on how the logger is configured. If the logger is writing
        to a file, :py:meth:`query()` returns the last value written to the file.
        If filename is *None*, then :py:meth:`query()` returns the value of the quantity computed at the current timestep.

        Examples::

            logdata = logger.query('pair_lj_energy')
            log = analyze.log(filename=None, quantities=['potential_energy'], period=1)
            U = log.query('potential_energy')

        """
        use_cache=True;
        if self.filename == "":
            use_cache = False;

        return self.cpp_analyzer.getQuantity(quantity, hoomd.context.current.system.getCurrentTimeStep(), use_cache);

    def register_callback(self, name, callback):
        R""" Register a callback to produce a logged quantity.

        Args:
            name (str): Name of the quantity
            callback (callable): A python callable object (i.e. a lambda, function, or class that implements __call__)

        The callback method must take a single argument, the current timestep, and return a single floating point value to
        be logged.

        Note:
            One callback can query the value of another, but logged quantities are evaluated in order from left to right.

        Examples::

            logger = analyze.log(filename='log.dat', quantities=['my_quantity', 'cosm'], period=100)
            logger.register_callback('my_quantity', lambda timestep: timestep**2)
            logger.register_callback('cosm', lambda timestep: math.cos(logger.query('my_quantity')))

        """
        self.cpp_analyzer.registerCallback(name, callback);

    ## \internal
    # \brief Re-registers all computes and updaters with the logger
    def update_quantities(self):
        # remove all registered quantities
        self.cpp_analyzer.removeAll();

        # re-register all computes and updater
        hoomd.context.current.system.registerLogger(self.cpp_analyzer);


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

    :py:class:`analyze.msd` can be given any number of groups of particles. Every *period* time steps, it calculates the mean squared
    displacement of each group (referenced to the particle positions at the time step the command is issued at) and prints
    the calculated values out to a file.

    The mean squared displacement (MSD) for each group is calculated as:

    .. math::
        \langle |\vec{r} - \vec{r}_0|^2 \rangle

    and values are correspondingly written in units of distance squared.

    The file format is the same convenient delimited format used by :py:class`analyze.log`.

    :py:class:`analyze.msd` is capable of appending to an existing msd file (the default setting) for use in restarting in long jobs.
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
    tab-delimited spreadsheet. The delimiter can be changed to any string using :py:meth`set_params()`.

    The *header_prefix* can be used in a number of ways. It specifies a simple string that
    will be printed before the header line of the output file. One handy way to use this
    is to specify header_prefix='#' so that ``gnuplot`` will ignore the header line
    automatically. Another use-case would be to specify a descriptive line containing
    details of the current run. Examples of each of these cases are given above.

    If *r0_file* is left at the default of None, then the current state of the system at the execution of the
    analyze.msd command is used to initialize :math:`\vec{r}_0`.
    """

    def __init__(self, filename, groups, period, header_prefix='', r0_file=None, overwrite=False, phase=-1):
        hoomd.util.print_status_line();

        # initialize base class
        _analyzer.__init__(self);

        # create the c++ mirror class
        self.cpp_analyzer = _hoomd.MSDAnalyzer(hoomd.context.current.system_definition, filename, header_prefix, overwrite);
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

class callback(_analyzer):
    R""" Callback analyzer.

    Args:
        callback (callable): The python callback object
        period (int): The callback is called every \a period time steps
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where (step + phase) % period == 0.

    Create an analyzer that runs a given python callback method at a defined period.

    Examples::

        def my_callback(timestep):
          print(timestep)

        analyze.callback(callback = my_callback, period = 100)
    """
    def __init__(self, callback, period, phase=-1):
        hoomd.util.print_status_line();

        # initialize base class
        _analyzer.__init__(self);

        # create the c++ mirror class
        self.cpp_analyzer = _hoomd.CallbackAnalyzer(hoomd.context.current.system_definition, callback)
        self.setupAnalyzer(period, phase);
