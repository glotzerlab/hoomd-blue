# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Dihedral potentials.

Dihedrals add forces between specified quadruplets of particles and are typically used to
model rotation about chemical bonds.

By themselves, dihedrals that have been specified in an input file do nothing. Only when you
specify an dihedral force (i.e. dihedral.harmonic), are forces actually calculated between the
listed particles.

Important:
There are multiple conventions pertaining to the dihedral angle (phi) in the literature. HOOMD
utilizes the convention shown in the following figure, where vectors are defined from the central
particles to the outer particles. These vectors correspond to a stretched state (phi=180 deg)
when they are anti-parallel and a compact state (phi=0 deg) when they are parallel.

.. image:: dihedral-angle-definition.png
    :width: 400 px
    :align: center
    :alt: Dihedral angle definition
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force
import hoomd;

import math;
import sys;

class coeff:
    R""" Defines dihedral coefficients.

    The coefficients for all dihedral force are specified using this class. Coefficients are
    specified per dihedral type.

    There are two ways to set the coefficients for a particular dihedral force.
    The first way is to save the dihedral force in a variable and call :py:meth:`set()` directly.
    See below for an example of this.

    The second method is to build the :py:class:`coeff` class first and then assign it to the
    dihedral force. There are some advantages to this method in that you could specify a
    complicated set of dihedral force coefficients in a separate python file and import
    it into your job script.

    Examples::

        my_coeffs = dihedral.coeff();
        my_dihedral_force.dihedral_coeff.set('polymer', k=330.0, r=0.84)
        my_dihedral_force.dihedral_coeff.set('backbone', k=330.0, r=0.84)
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
        R""" Sets parameters for dihedral types.

        Args:
            type (str): Type of dihedral, or list of types
            coeffs: Named coefficients (see below for examples)

        Calling :py:meth:`set()` results in one or more parameters being set for a dihedral type. Types are identified
        by name, and parameters are also added by name. Which parameters you need to specify depends on the dihedral
        force you are setting these coefficients for, see the corresponding documentation.

        All possible dihedral types as defined in the simulation box must be specified before executing run().
        You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
        dihedral types that do not exist in the simulation. This can be useful in defining a force field for many
        different types of dihedrals even when some simulations only include a subset.

        To set the same coefficients between many particle types, provide a list of type names instead of a single
        one. All types in the list will be set to the same parameters.

        Examples::

            my_dihedral_force.dihedral_coeff.set('polymer', k=330.0, r0=0.84)
            my_dihedral_force.dihedral_coeff.set('backbone', k=1000.0, r0=1.0)
            my_dihedral_force.dihedral_coeff.set(['dihedralA','dihedralB'], k=100, r0=0.0)

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
            hoomd.context.msg.error("Cannot verify dihedral coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getDihedralData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getDihedralData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                hoomd.context.msg.error("Dihedral type " +str(type) + " not found in dihedral coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    hoomd.context.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the dihedral force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                hoomd.context.msg.error("Dihedral type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single dihedral force coefficient
    # \detail
    # \param type Name of dihedral type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            hoomd.context.msg.error("Bug detected in force.coeff. Please report\n");
            raise RuntimeError("Error setting dihedral coeff");

        return self.values[type][coeff_name];

    ## \internal
    # \brief Return metadata
    def get_metadata(self):
        return self.values

class harmonic(force._force):
    R""" Harmonic dihedral potential.

    :py:class:`harmonic` specifies a harmonic dihedral potential energy between every defined dihedral
    quadruplet of particles in the simulation:

    .. math::

        V(r) = \frac{1}{2}k \left( 1 + d \cos\left(n * \phi(r) - \phi_0 \right) \right)

    where :math:`\phi` is angle between two sides of the dihedral.

    Coefficients:

    - :math:`k` - strength of force (in energy units)
    - :math:`d` - sign factor (unitless)
    - :math:`n` - angle scaling factor (unitless)
    - :math:`\phi_0` - phase shift  ``phi_0`` (in radians) - *optional*: defaults to 0.0

    Coefficients :math:`k`, :math:`d`, :math:`n` must be set for each type of dihedral in the simulation using
    :py:meth:`dihedral_coeff.set() <coeff.set()>`.

    Examples::

        harmonic.dihedral_coeff.set('phi-ang', k=30.0, d=-1, n=3)
        harmonic.dihedral_coeff.set('psi-ang', k=100.0, d=1, n=4, phi_0=math.pi/2)
    """
    def __init__(self):
        hoomd.util.print_status_line();
        # check that some dihedrals are defined
        if hoomd.context.current.system_definition.getDihedralData().getNGlobal() == 0:
            hoomd.context.msg.error("No dihedrals are defined.\n");
            raise RuntimeError("Error creating dihedral forces");

        # initialize the base class
        force._force.__init__(self);

        self.dihedral_coeff = coeff();

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.HarmonicDihedralForceCompute(hoomd.context.current.system_definition);
        else:
            self.cpp_force = _md.HarmonicDihedralForceComputeGPU(hoomd.context.current.system_definition);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        self.required_coeffs = ['k', 'd', 'n', 'phi_0'];
        self.dihedral_coeff.set_default_coeff('phi_0', 0.0);

    ## \internal
    # \brief Update coefficients in C++
    def update_coeffs(self):
        coeff_list = self.required_coeffs;
        # check that the force coefficients are valid
        if not self.dihedral_coeff.verify(coeff_list):
           hoomd.context.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getDihedralData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getDihedralData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.dihedral_coeff.get(type_list[i], name);

            self.cpp_force.setParams(i, coeff_dict['k'], coeff_dict['d'], coeff_dict['n'], coeff_dict['phi_0']);

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['dihedral_coeff'] = self.dihedral_coeff
        return data

def _table_eval(theta, V, T, width):
      dth = (2*math.pi) / float(width-1);
      i = int(round((theta+math.pi)/dth))
      return (V[i], T[i])

class table(force._force):
    R""" Tabulated dihedral potential.

    Args:
        width (int): Number of points to use to interpolate V and T (see documentation above)
        name (str): Name of the force instance

    :py:class:`table` specifies that a tabulated dihedral force should be applied to every define dihedral.

    :math:`T_{\mathrm{user}}(\theta)` and :math:`V_{\mathrm{user}}(\theta)` are evaluated on *width* grid points between
    :math:`-\pi` and :math:`\pi`. Values are interpolated linearly between grid points.
    For correctness, you must specify the derivative of the potential with respect to the dihedral angle,
    defined by: :math:`T = -\frac{\partial V}{\partial \theta}`.

    Parameters:

    - :math:`T_{\mathrm{user}}(\theta)` and :math:`V_{\mathrm{user}} (\theta)` - evaluated by ``func`` (see example)
    - coefficients passed to `func` - `coeff` (see example)

    .. rubric:: Set table from a given function

    When you have a functional form for V and T, you can enter that
    directly into python. :py:class:`table` will evaluate the given function over *width* points between :math:`-\pi` and :math:`\pi`
    and use the resulting values in the table::

        def harmonic(theta, kappa, theta0):
           V = 0.5 * kappa * (theta-theta0)**2;
           F = -kappa*(theta-theta0);
           return (V, F)

        dtable = dihedral.table(width=1000)
        dtable.dihedral_coeff.set('dihedral1', func=harmonic, coeff=dict(kappa=330, theta_0=0.0))
        dtable.dihedral_coeff.set('dihedral2', func=harmonic,coeff=dict(kappa=30, theta_0=1.0))

    .. rubric:: Set a table from a file

    When you have no function for for *V* or *T*, or you otherwise have the data listed in a file, dihedral.table can use the given
    values directly. You must first specify the number of rows in your tables when initializing :py:class:`table`. Then use
    :py:meth:`set_from_file()` to read the file.

        dtable = dihedral.table(width=1000)
        dtable.set_from_file('polymer', 'dihedral.dat')

    """
    def __init__(self, width, name=None):
        hoomd.util.print_status_line();

        # initialize the base class
        force._force.__init__(self, name);


        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.TableDihedralForceCompute(hoomd.context.current.system_definition, int(width), self.name);
        else:
            self.cpp_force = _md.TableDihedralForceComputeGPU(hoomd.context.current.system_definition, int(width), self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient matrix
        self.dihedral_coeff = coeff();

        # stash the width for later use
        self.width = width;

    def update_dihedral_table(self, atype, func, coeff):
        # allocate arrays to store V and F
        Vtable = _hoomd.std_vector_scalar();
        Ttable = _hoomd.std_vector_scalar();

        # calculate dth
        dth = 2.0*math.pi / float(self.width-1);

        # evaluate each point of the function
        for i in range(0, self.width):
            theta = -math.pi+dth * i;
            (V,T) = func(theta, **coeff);

            # fill out the tables
            Vtable.append(V);
            Ttable.append(T);

        # pass the tables on to the underlying cpp compute
        self.cpp_force.setTable(atype, Vtable, Ttable);


    def update_coeffs(self):
        # check that the dihedral coefficients are valid
        if not self.dihedral_coeff.verify(["func", "coeff"]):
            hoomd.context.msg.error("Not all dihedral coefficients are set for dihedral.table\n");
            raise RuntimeError("Error updating dihedral coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getDihedralData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getDihedralData().getNameByType(i));


        # loop through all of the unique type dihedrals and evaluate the table
        for i in range(0,ntypes):
            func = self.dihedral_coeff.get(type_list[i], "func");
            coeff = self.dihedral_coeff.get(type_list[i], "coeff");

            self.update_dihedral_table(i, func, coeff);

    def set_from_file(self, dihedralname, filename):
        R"""  Set a dihedral pair interaction from a file.

        Args:
            dihedralname (str): Name of dihedral
            filename (str): Name of the file to read

        The provided file specifies V and F at equally spaced theta values.

        Example::

            #t  V    T
            -3.141592653589793 2.0 -3.0
            -1.5707963267948966 3.0 -4.0
            0.0 2.0 -3.0
            1.5707963267948966 3.0 -4.0
            3.141592653589793 2.0 -3.0

        Note:
            The theta values are not used by the code.  It is assumed that a table that has N rows will start at :math:`-\pi`, end at :math:`\pi`
            and that :math:`\delta \theta = 2\pi/(N-1)`. The table is read
            directly into the grid points used to evaluate :math:`T_{\mathrm{user}}(\theta)` and :math:`V_{\mathrm{user}}(\theta)`.

        """
        hoomd.util.print_status_line();

        # open the file
        f = open(filename);

        theta_table = [];
        V_table = [];
        T_table = [];

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
                hoomd.context.msg.error("dihedral.table: file must have exactly 3 columns\n");
                raise RuntimeError("Error reading table file");

            # append to the tables
            theta_table.append(values[0]);
            V_table.append(values[1]);
            T_table.append(values[2]);

        # validate input
        if self.width != len(T_table):
            hoomd.context.msg.error("dihedral.table: file must have exactly " + str(self.width) + " rows\n");
            raise RuntimeError("Error reading table file");


        # check for even spacing
        dth = 2.0*math.pi / float(self.width-1);
        for i in range(0,self.width):
            theta =  -math.pi+dth * i;
            if math.fabs(theta - theta_table[i]) > 1e-3:
                hoomd.context.msg.error("dihedral.table: theta must be monotonically increasing and evenly spaced, going from -pi to pi\n");
                hoomd.context.msg.error("row: " + str(i) + " expected: " + str(theta) + " got: " + str(theta_table[i]) + "\n");

        hoomd.util.quiet_status();
        self.dihedral_coeff.set(dihedralname, func=_table_eval, coeff=dict(V=V_table, T=T_table, width=self.width))
        hoomd.util.unquiet_status();

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['dihedral_coeff'] = self.dihedral_coeff
        return data

class opls(force._force):
    R""" OPLS dihedral force

    :py:class:`opls` specifies an OPLS-style dihedral potential energy between every defined dihedral.

    .. math::

        V(r) = \frac{1}{2}k_1 \left( 1 + \cos\left(\phi \right) \right) + \frac{1}{2}k_2 \left( 1 - \cos\left(2 \phi \right) \right)
               + \frac{1}{2}k_3 \left( 1 + \cos\left(3 \phi \right) \right) + \frac{1}{2}k_4 \left( 1 - \cos\left(4 \phi \right) \right)

    where :math:`\phi` is the angle between two sides of the dihedral and :math:`k_n` are the force coefficients
    in the Fourier series (in energy units).

    :math:`k_1`, :math:`k_2`, :math:`k_3`, and :math:`k_4` must be set for each type of dihedral in the simulation using
    :py:meth:`dihedral_coeff.set() <coeff.set()>`.

    Example::

        opls_di.dihedral_coeff.set('dihedral1', k1=30.0, k2=15.5, k3=2.2, k4=23.8)

    """
    def __init__(self):
        hoomd.util.print_status_line();
        # check that some dihedrals are defined
        if hoomd.context.current.system_definition.getDihedralData().getNGlobal() == 0:
            hoomd.context.msg.error("No dihedrals are defined.\n");
            raise RuntimeError("Error creating opls dihedrals");

        # initialize the base class
        force._force.__init__(self);

        self.dihedral_coeff = coeff();

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.OPLSDihedralForceCompute(hoomd.context.current.system_definition);
        else:
            self.cpp_force = _md.OPLSDihedralForceComputeGPU(hoomd.context.current.system_definition);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        self.required_coeffs = ['k1', 'k2', 'k3', 'k4'];

    ## \internal
    # \brief Update coefficients in C++
    def update_coeffs(self):
        coeff_list = self.required_coeffs;
        # check that the force coefficients are valid
        if not self.dihedral_coeff.verify(coeff_list):
           hoomd.context.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getDihedralData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getDihedralData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.dihedral_coeff.get(type_list[i], name);

            self.cpp_force.setParams(i, coeff_dict['k1'], coeff_dict['k2'], coeff_dict['k3'], coeff_dict['k4']);

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['dihedral_coeff'] = self.dihedral_coeff
        return data
