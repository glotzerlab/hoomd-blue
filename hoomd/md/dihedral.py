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
from hoomd.md.force import _Force
from hoomd.parameterdicts import TypeParameterDict
from hoomd.typeparam import TypeParameter
import hoomd

import math


class _Dihedral(_Force):
    """Constructs the dihedral bond potential.

    Note:
        :py:class:`_Dihedral`is the base class for all dihedral potentials.
        Users should not instantiate this class directly. Dihedral forces
        documented here are available to all MD integrators.
    
    A dihedral bond in hoomd reflects a PotentialBond in c++. It is responsible for all
    high-level management that happens behind the scenes for hoomd writers.
    1) The instance of the c++ dihedral bond force itself is tracked and added to the
    System
    2) methods are provided for disabling the force from being added to the net
    force on each particle
    """
    def attach(self, simulation):
        '''initialize the reflected c++ class'''
        if simulation.state._cpp_sys_def.getDihedralData().getNGlobal() == 0:
            simulation.device.cpp_msg.warning("No dihedrals are defined.\n")

        # create the c++ mirror class
        if not simulation.device.cpp_exec_conf.isCUDAEnabled():
            cpp_class = getattr(_md, self._cpp_class_name)
        else:
            cpp_class = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_class(simulation.state._cpp_sys_def)
        super().attach(simulation)


class Harmonic(_Dihedral):
    R""" Harmonic dihedral potential.

    :py:class:`Harmonic` specifies a harmonic dihedral potential energy between
    every defined dihedral quadruplet of particles in the simulation:

    .. math::

        V(r) = \frac{1}{2}k \left( 1 + d \cos\left(n * \phi(r) -
               \phi_0 \right) \right)

    where :math:`\phi` is angle between two sides of the dihedral.

    Attributes:
        params (TypeParameter[``dihedral type``, dict]):
            The parameter of the harmonic bonds for each particle type. 
            The dictionary has the following keys: 

            * ``k`` (`float`, **required**) - potential constant 
              (in units of energy)

            * ``d`` (`float`, **required**) - sign factor 
              (unitless)

            * ``n`` (`float`, **required**) - angle scalinf factor 
              (unitless)

            * ``phi0`` (`float`, **required**) - phase shift 
              (in units of radians)

    Examples::

        harmonic = dihedral.Harmonic()
        harmonic.params['polymer'] = dict(k=3.0, d=-1, n=3, phi0=0)
        harmonic.params['backbone'] = dict(k=100.0, d=1, n=4, phi0=math.pi/2)

    """

    _cpp_class_name = "HarmonicDihedralForceCompute"

    def __init__(self):
        params = TypeParameter('params', 'dihedral_types',
                               TypeParameterDict(k=float, d=float,
                                                 n=float, phi0=float,
                                                 len_keys=1))
        self._add_typeparam(params)


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
    - coefficients passed to ``func`` - ``coeff`` (see example)

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

        # initialize the base class
        force._force.__init__(self, name);


        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
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
            hoomd.context.current.device.cpp_msg.error("Not all dihedral coefficients are set for dihedral.table\n");
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
                hoomd.context.current.device.cpp_msg.error("dihedral.table: file must have exactly 3 columns\n");
                raise RuntimeError("Error reading table file");

            # append to the tables
            theta_table.append(values[0]);
            V_table.append(values[1]);
            T_table.append(values[2]);

        # validate input
        if self.width != len(T_table):
            hoomd.context.current.device.cpp_msg.error("dihedral.table: file must have exactly " + str(self.width) + " rows\n");
            raise RuntimeError("Error reading table file");


        # check for even spacing
        dth = 2.0*math.pi / float(self.width-1);
        for i in range(0,self.width):
            theta =  -math.pi+dth * i;
            if math.fabs(theta - theta_table[i]) > 1e-3:
                hoomd.context.current.device.cpp_msg.error("dihedral.table: theta must be monotonically increasing and evenly spaced, going from -pi to pi\n");
                hoomd.context.current.device.cpp_msg.error("row: " + str(i) + " expected: " + str(theta) + " got: " + str(theta_table[i]) + "\n");

        self.dihedral_coeff.set(dihedralname, func=_table_eval, coeff=dict(V=V_table, T=T_table, width=self.width))

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['dihedral_coeff'] = self.dihedral_coeff
        return data


class OPLS(_Dihedral):
    R""" OPLS dihedral force

    :py:class:`OPLS` specifies an OPLS-style dihedral potential energy between
    every defined dihedral.

    .. math::

        V(r) = \frac{1}{2}k_1 \left( 1 + \cos\left(\phi \right) \right) +
               \frac{1}{2}k_2 \left( 1 - \cos\left(2 \phi \right) \right)
               + \frac{1}{2}k_3 \left( 1 + \cos\left(3 \phi \right) \right) +
               \frac{1}{2}k_4 \left( 1 - \cos\left(4 \phi \right) \right)

    where :math:`\phi` is the angle between two sides of the dihedral and
    :math:`k_n` are the force coefficients in the Fourier series (in energy
    units).

    Attributes:
        params (TypeParameter[``dihedral type``, dict]):
            The parameter of the OPLS bonds for each particle type. 
            The dictionary has the following keys: 

            * ``k1`` (`float`, **required**) -  force constant of the first term
              (in units of energy)

            * ``k2`` (`float`, **required**) -  force constant of the second term
              (in units of energy)

            * ``k3`` (`float`, **required**) -  force constant of the third term
              (in units of energy)

            * ``k4`` (`float`, **required**) -  force constant of the fourth term
              (in units of energy)

    Examples::

        opls = dihedral.OPLS()
        opls.params['backbone'] = dict(k1=1.0, k2=1.0, k3=1.0, k4=1.0)

    """

    _cpp_class_name = "OPLSDihedralForceCompute"

    def __init__(self):
        # check that some dihedrals are defined
        params = TypeParameter('params', 'dihedral_types',
                               TypeParameterDict(k1=float, k2=float,
                                                 k3=float, k4=float,
                                                 len_keys=1))
        self._add_typeparam(params)
