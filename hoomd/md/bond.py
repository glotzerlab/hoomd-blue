# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Bond potentials.

Bonds add forces between specified pairs of particles and are typically used to
model chemical bonds between two particles.

By themselves, bonds that have been specified in an initial configuration do nothing. Only when you
specify an bond force (i.e. bond.harmonic), are forces actually calculated between the
listed particles.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force;
import hoomd;

import math;
import sys;

class coeff:
    R""" Define bond coefficients.

    The coefficients for all bond potentials are specified using this class. Coefficients are
    specified per bond type.

    There are two ways to set the coefficients for a particular bond potential.
    The first way is to save the bond potential in a variable and call :py:meth:`set()` directly.
    See below for an example of this.

    The second method is to build the coeff class first and then assign it to the
    bond potential. There are some advantages to this method in that you could specify a
    complicated set of bond potential coefficients in a separate python file and import
    it into your job script.

    Example::

        my_coeffs = hoomd.md.bond.coeff();
        my_bond_force.bond_coeff.set('polymer', k=330.0, r=0.84)
        my_bond_force.bond_coeff.set('backbone', k=330.0, r=0.84)

    """

    ## \internal
    # \brief Initializes the class
    # \details
    # The main task to be performed during initialization is just to init some variables
    # \param self Python required class instance variable
    def __init__(self):
        self.values = {};
        self.default_coeff = {}

    ## \var values
    # \internal
    # \brief Contains the vector of set values in a dictionary

    ## \var default_coeff
    # \internal
    # \brief default_coeff['coeff'] lists the default value for \a coeff, if it is set

    ## \internal
    # \brief Sets a default value for a given coefficient
    # \details
    # \param name Name of the coefficient to for which to set the default
    # \param value Default value to set
    #
    # Some coefficients have reasonable default values and the user should not be burdened with typing them in
    # all the time. set_default_coeff() sets
    def set_default_coeff(self, name, value):
        self.default_coeff[name] = value;

    def set(self, type, **coeffs):
        R""" Sets parameters for bond types.

        Args:
            type (str): Type of bond (or a list of type names)
            coeffs: Named coefficients (see below for examples)

        Calling :py:meth:`set()` results in one or more parameters being set for a bond type. Types are identified
        by name, and parameters are also added by name. Which parameters you need to specify depends on the bond
        potential you are setting these coefficients for, see the corresponding documentation.

        All possible bond types as defined in the simulation box must be specified before executing run().
        You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
        bond types that do not exist in the simulation. This can be useful in defining a potential field for many
        different types of bonds even when some simulations only include a subset.

        Examples::

            my_bond_force.bond_coeff.set('polymer', k=330.0, r0=0.84)
            my_bond_force.bond_coeff.set('backbone', k=1000.0, r0=1.0)
            my_bond_force.bond_coeff.set(['bondA','bondB'], k=100, r0=0.0)

        Note:
            Single parameters can be updated. If both ``k`` and ``r0`` have already been set for a particle type,
            then executing ``coeff.set('polymer', r0=1.0)`` will update the value of ``r0`` and leave the other
            parameters as they were previously set.

        """
        hoomd.util.print_status_line();

        # listify the input
        type = hoomd.util.listify(type)

        for typei in type:
            self.set_single(typei, coeffs);

    ## \internal
    # \brief Sets a single parameter
    def set_single(self, type, coeffs):
        type = str(type);

        # create the type identifier if it hasn't been created yet
        if (not type in self.values):
            self.values[type] = {};

        # update each of the values provided
        if len(coeffs) == 0:
            hoomd.context.msg.error("No coefficients specified\n");
        for name, val in coeffs.items():
            self.values[type][name] = val;

        # set the default values
        for name, val in self.default_coeff.items():
            # don't override a coeff if it is already set
            if not name in self.values[type]:
                self.values[type][name] = val;

    ## \internal
    # \brief Verifies that all values are set
    # \details
    # \param self Python required self variable
    # \param required_coeffs list of required variables
    #
    # This can only be run after the system has been initialized
    def verify(self, required_coeffs):
        # first, check that the system has been initialized
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot verify bond coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getBondData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getBondData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                hoomd.context.msg.error("Bond type " +str(type) + " not found in bond coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    hoomd.context.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the bond force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                hoomd.context.msg.error("Bond type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %bond %force coefficient
    # \detail
    # \param type Name of bond type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            hoomd.context.msg.error("Bug detected in force.coeff. Please report\n");
            raise RuntimeError("Error setting bond coeff");

        return self.values[type][coeff_name];

    ## \internal
    # \brief Return metadata
    def get_metadata(self):
        return self.values

## \internal
# \brief Base class for bond potentials
#
# A bond in hoomd reflects a PotentialBond in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd
# writers. 1) The instance of the c++ bond force itself is tracked and added to the
# System 2) methods are provided for disabling the force from being added to the
# net force on each particle
class _bond(force._force):
    ## \internal
    # \brief Constructs the bond potential
    #
    # \param name name of the bond potential instance
    #
    # Initializes the cpp_force to None.
    # If specified, assigns a name to the instance
    # Assigns a name to the force in force_name;
    def __init__(self, name=None):
        # initialize the base class
        force._force.__init__(self, name);

        self.cpp_force = None;

        # setup the coefficient vector
        self.bond_coeff = coeff();

        self.enabled = True;

    def update_coeffs(self):
        coeff_list = self.required_coeffs;
        # check that the force coefficients are valid
        if not self.bond_coeff.verify(coeff_list):
           hoomd.context.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getBondData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getBondData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.bond_coeff.get(type_list[i], name);

            param = self.process_coeff(coeff_dict);
            self.cpp_force.setParams(i, param);

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['bond_coeff'] = self.bond_coeff
        return data

class harmonic(_bond):
    R""" Harmonic bond potential.

    Args:
        name (str): Name of the bond instance.

    :py:class:`harmonic` specifies a harmonic potential energy between the two particles in each defined bond.

    .. math::

        V(r) = \frac{1}{2} k \left( r - r_0 \right)^2

    where :math:`\vec{r}` is the vector pointing from one particle to the other in the bond.

    Coefficients:

    - :math:`k` - force constant ``k`` (in units of energy/distance^2)
    - :math:`r_0` - bond rest length ``r0`` (in distance units)

    Example::

        harmonic = bond.harmonic(name="mybond")
        harmonic.bond_coeff.set('polymer', k=330.0, r0=0.84)

    """
    def __init__(self,name=None):
        hoomd.util.print_status_line();

        # initialize the base class
        _bond.__init__(self);


        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.PotentialBondHarmonic(hoomd.context.current.system_definition,self.name);
        else:
            self.cpp_force = _md.PotentialBondHarmonicGPU(hoomd.context.current.system_definition,self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = ['k','r0'];

    def process_coeff(self, coeff):
        k = coeff['k'];
        r0 = coeff['r0'];

        # set the parameters for the appropriate type
        return _hoomd.make_scalar2(k, r0);

class fene(_bond):
    R""" FENE bond potential.

    Args:
        name (str): Name of the bond instance.

    :py:class:`fene` specifies a FENE potential energy between the two particles in each defined bond.

    .. math::

        V(r) = - \frac{1}{2} k r_0^2 \ln \left( 1 - \left( \frac{r - \Delta}{r_0} \right)^2 \right) + V_{\mathrm{WCA}}(r)

    where :math:`\vec{r}` is the vector pointing from one particle to the other in the bond,
    :math:`\Delta = (d_i + d_j)/2 - 1`, :math:`d_i` is the diameter of particle :math:`i`, and

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{WCA}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r - \Delta} \right)^{12} - \left( \frac{\sigma}{r - \Delta} \right)^{6} \right]  + \varepsilon & r-\Delta < 2^{\frac{1}{6}}\sigma\\
                   = & 0          & r-\Delta \ge 2^{\frac{1}{6}}\sigma
        \end{eqnarray*}

    Coefficients:

    - :math:`k` - attractive force strength ``k`` (in units of energy/distance^2)
    - :math:`r_0` - size parameter ``r0`` (in distance units)
    - :math:`\varepsilon` - repulsive force strength ``epsilon`` (in energy units)
    - :math:`\sigma` - repulsive force interaction distance ``sigma`` (in distance units)

    Examples::

        fene = bond.fene()
        fene.bond_coeff.set('polymer', k=30.0, r0=1.5, sigma=1.0, epsilon= 2.0)
        fene.bond_coeff.set('backbone', k=100.0, r0=1.0, sigma=1.0, epsilon= 2.0)

    """
    def __init__(self, name=None):
        hoomd.util.print_status_line();


        # initialize the base class
        _bond.__init__(self, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.PotentialBondFENE(hoomd.context.current.system_definition,self.name);
        else:
            self.cpp_force = _md.PotentialBondFENEGPU(hoomd.context.current.system_definition,self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = ['k','r0','epsilon','sigma'];

    def process_coeff(self, coeff):
        k = coeff['k'];
        r0 = coeff['r0'];
        lj1 = 4.0 * coeff['epsilon'] * math.pow(coeff['sigma'], 12.0);
        lj2 = 4.0 * coeff['epsilon'] * math.pow(coeff['sigma'], 6.0);
        return _hoomd.make_scalar4(k, r0, lj1, lj2);




def _table_eval(r, rmin, rmax, V, F, width):
      dr = (rmax - rmin) / float(width-1);
      i = int(round((r - rmin)/dr))
      return (V[i], F[i])

class table(force._force):
    R""" Tabulated bond potential.

    Args:
        width (int): Number of points to use to interpolate V and F
        name (str): Name of the potential instance

    :py:class:`table` specifies that a tabulated bond potential should be applied between the two particles in each
    defined bond.

    The force :math:`\vec{F}` is (in force units) and the potential :math:`V(r)` is (in energy units):

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}(\vec{r})     = & 0                           & r < r_{\mathrm{min}} \\
                             = & F_{\mathrm{user}}(r)\hat{r} & r < r_{\mathrm{max}} \\
                             = & 0                           & r \ge r_{\mathrm{max}} \\
                             \\
        V(r)       = & 0                    & r < r_{\mathrm{min}} \\
                   = & V_{\mathrm{user}}(r) & r < r_{\mathrm{max}} \\
                   = & 0                    & r \ge r_{\mathrm{max}} \\
        \end{eqnarray*}

    where :math:`\vec{r}` is the vector pointing from one particle to the other in the bond.  Care should be taken to
    define the range of the bond so that it is not possible for the distance between two bonded particles to be outside the
    specified range.  On the CPU, this will throw an error.  On the GPU, this will throw an error if GPU error checking is enabled.

    :math:`F_{\mathrm{user}}(r)` and :math:`V_{\mathrm{user}}(r)` are evaluated on *width* grid points between
    :math:`r_{\mathrm{min}}` and :math:`r_{\mathrm{max}}`. Values are interpolated linearly between grid points.
    For correctness, you must specify the force defined by: :math:`F = -\frac{\partial V}{\partial r}`

    The following coefficients must be set for each bond type:

    - :math:`F_{\mathrm{user}}(r)` and :math:`V_{\mathrm{user}}(r)` - evaluated by ``func`` (see example)
    - coefficients passed to `func` - ``coeff`` (see example)
    - :math:`r_{\mathrm{min}}` - ``rmin`` (in distance units)
    - :math:`r_{\mathrm{max}}` - ``rmax`` (in distance units)

    The table *width* is set once when bond.table is specified.
    There are two ways to specify the other parameters.

    .. rubric:: Set table from a given function

    When you have a functional form for V and F, you can enter that
    directly into python. :py:class:`table` will evaluate the given function over *width* points between *rmin* and *rmax*
    and use the resulting values in the table::

        def harmonic(r, rmin, rmax, kappa, r0):
           V = 0.5 * kappa * (r-r0)**2;
           F = -kappa*(r-r0);
           return (V, F)

        btable = bond.table(width=1000)
        btable.bond_coeff.set('bond1', func=harmonic, rmin=0.2, rmax=5.0, coeff=dict(kappa=330, r0=0.84))
        btable.bond_coeff.set('bond2', func=harmonic, rmin=0.2, rmax=5.0, coeff=dict(kappa=30, r0=1.0))

    .. rubric:: Set a table from a file

    When you have no function for for *V* or *F*, or you otherwise have the data listed in a file, :py:class:`table` can use the given
    values directly. You must first specify the number of rows in your tables when initializing bond.table. Then use
    :py:meth:`set_from_file()` to read the file::

        btable = bond.table(width=1000)
        btable.set_from_file('polymer', 'btable.file')

    Note:
        For potentials that diverge near r=0, make sure to set ``rmin`` to a reasonable value. If a potential does
        not diverge near r=0, then a setting of ``rmin=0`` is valid.

    Note:
        Ensure that ``rmin`` and ``rmax`` cover the range of possible bond lengths. When gpu error checking is on, a error will
        be thrown if a bond distance is outside than this range.
    """
    def __init__(self, width, name=None):
        hoomd.util.print_status_line();

        # initialize the base class
        force._force.__init__(self, name);


        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.BondTablePotential(hoomd.context.current.system_definition, int(width), self.name);
        else:
            self.cpp_force = _md.BondTablePotentialGPU(hoomd.context.current.system_definition, int(width), self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficients matrix
        self.bond_coeff = coeff();

        # stash the width for later use
        self.width = width;

    def update_bond_table(self, btype, func, rmin, rmax, coeff):
        # allocate arrays to store V and F
        Vtable = _hoomd.std_vector_scalar();
        Ftable = _hoomd.std_vector_scalar();

        # calculate dr
        dr = (rmax - rmin) / float(self.width-1);

        # evaluate each point of the function
        for i in range(0, self.width):
            r = rmin + dr * i;
            (V,F) = func(r, rmin, rmax, **coeff);

            # fill out the tables
            Vtable.append(V);
            Ftable.append(F);

        # pass the tables on to the underlying cpp compute
        self.cpp_force.setTable(btype, Vtable, Ftable, rmin, rmax);


    def update_coeffs(self):
        # check that the bond coefficients are valid
        if not self.bond_coeff.verify(["func", "rmin", "rmax", "coeff"]):
            hoomd.context.msg.error("Not all bond coefficients are set for bond.table\n");
            raise RuntimeError("Error updating bond coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getBondData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getBondData().getNameByType(i));


        # loop through all of the unique type bonds and evaluate the table
        for i in range(0,ntypes):
            func = self.bond_coeff.get(type_list[i], "func");
            rmin = self.bond_coeff.get(type_list[i], "rmin");
            rmax = self.bond_coeff.get(type_list[i], "rmax");
            coeff = self.bond_coeff.get(type_list[i], "coeff");

            self.update_bond_table(i, func, rmin, rmax, coeff);

    def set_from_file(self, bondname, filename):
        R""" Set a bond pair interaction from a file.

        Args:
            bondname (str): Name of bond
            filename (str): Name of the file to read

        The provided file specifies V and F at equally spaced r values.
        Example::

            #r  V    F
            1.0 2.0 -3.0
            1.1 3.0 -4.0
            1.2 2.0 -3.0
            1.3 1.0 -2.0
            1.4 0.0 -1.0
            1.5 -1.0 0.0

        The first r value sets ``rmin``, the last sets ``rmax``. Any line with # as the first non-whitespace character is
        is treated as a comment. The ``r`` values must monotonically increase and be equally spaced. The table is read
        directly into the grid points used to evaluate :math:`F_{\mathrm{user}}(r)` and :math:`V_{\mathrm{user}}(r)`.
        """
        hoomd.util.print_status_line();

        # open the file
        f = open(filename);

        r_table = [];
        V_table = [];
        F_table = [];

        # read in lines from the file
        for line in f.readlines():
            line = line.strip();

            # skip comment lines
            if line[0] == '#':
                continue;

            # split out the columns
            cols = line.split();
            values = [float(f) for f in cols];

            # validate the input
            if len(values) != 3:
                hoomd.context.msg.error("bond.table: file must have exactly 3 columns\n");
                raise RuntimeError("Error reading table file");

            # append to the tables
            r_table.append(values[0]);
            V_table.append(values[1]);
            F_table.append(values[2]);

        # validate input
        if self.width != len(r_table):
            hoomd.context.msg.error("bond.table: file must have exactly " + str(self.width) + " rows\n");
            raise RuntimeError("Error reading table file");

        # extract rmin and rmax
        rmin_table = r_table[0];
        rmax_table = r_table[-1];

        # check for even spacing
        dr = (rmax_table - rmin_table) / float(self.width-1);
        for i in range(0,self.width):
            r = rmin_table + dr * i;
            if math.fabs(r - r_table[i]) > 1e-3:
                hoomd.context.msg.error("bond.table: r must be monotonically increasing and evenly spaced\n");
                raise RuntimeError("Error reading table file");

        hoomd.util.quiet_status();
        self.bond_coeff.set(bondname, func=_table_eval, rmin=rmin_table, rmax=rmax_table, coeff=dict(V=V_table, F=F_table, width=self.width))
        hoomd.util.unquiet_status();
