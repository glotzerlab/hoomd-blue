# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" Angle potentials.

Angles add forces between specified triplets of particles and are typically used to
model chemical angles between two bonds.

By themselves, angles that have been specified in an initial configuration do nothing. Only when you
specify an angle force (i.e. angle.harmonic), are forces actually calculated between the
listed particles.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force
from hoomd.md.force import _Force
from hoomd.typeparam import TypeParameter
from hoomd.parameterdicts import TypeParameterDict
import hoomd

import math


class _Angle(_Force):
    def attach(self, simulation):
        # check that some angles are defined
        if simulation.state._cpp_sys_def.getAngleData().getNGlobal() == 0:
            simulation.device.cpp_msg.warning("No angles are defined.\n")

        # create the c++ mirror class
        if not simulation.device.cpp_exec_conf.isCUDAEnabled():
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(simulation.state._cpp_sys_def)

        super()._attach(simulation)


class Harmonic(_Angle):
    R""" Harmonic angle potential.

    The command angle.harmonic specifies a harmonic potential energy between
    every triplet of particles with an angle specified between them.

    .. math::

        V(\theta) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2

    where :math:`\theta` is the angle between the triplet of particles.

    Parameters:

    - :math:`\theta_0` - rest angle  ``t0`` (in radians)
    - :math:`k` - potential constant ``k`` (in units of energy/radians^2)

    Examples::

        harmonic = angle.harmonic()
        harmonic.params['polymer'] = dict(k=3.0, t0=0.7851)
        harmonic.params['backbone'] = dict(k=100.0, t0=1.0)

    """

    _cpp_class_name = 'HarmonicAngleForceCompute'

    def __init__(self):
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(t0=float, k=float, len_keys=1))
        self._add_typeparam(params)


class Cosinesq(_Angle):
    R""" Cosine squared angle potential.

    The command angle.cosinesq specifies a cosine squared potential energy
    between every triplet of particles with an angle specified between them.

    .. math::

        V(\theta) = \frac{1}{2} k \left( \cos\theta - \cos\theta_0 \right)^2

    where :math:`\theta` is the angle between the triplet of particles.
    This angle style is also known as g96, since they were used in the
    gromos96 force field. These are also the types of angles used with the
    coarse-grained MARTINI force field.

    Params:

    - :math:`\theta_0` - rest angle  ``t0`` (in radians)
    - :math:`k` - potential constant ``k`` (in units of energy)

    Parameters :math:`k` and :math:`\theta_0` must be set for each type of
    angle in the simulation.  Note that the value of :math:`k` for this angle
    potential is not comparable to the value of :math:`k` for harmonic angles,
    as they have different units.

    Examples::

        cosinesq = angle.cosinesq()
        cosinesq.angle_coeff.set('polymer', k=3.0, t0=0.7851)
        cosinesq.angle_coeff.set('backbone', k=100.0, t0=1.0)

    """

    _cpp_class_name = 'CosineSqAngleForceCompute'

    def __init__(self):
        params = TypeParameter('params', 'angle_types',
                               TypeParameterDict(t0=float, k=float, len_keys=1))
        self._add_typeparam(params)


def _table_eval(theta, V, T, width):
      dth = (math.pi) / float(width-1);
      i = int(round((theta)/dth))
      return (V[i], T[i])


class table(force._force):
    R""" Tabulated angle potential.

    Args:

        width (int): Number of points to use to interpolate V and F (see documentation above)
        name (str): Name of the force instance

    :py:class:`table` specifies that a tabulated  angle potential should be added to every bonded triple of particles
    in the simulation.

    The torque :math:`T` is (in units of force * distance) and the potential :math:`V(\theta)` is (in energy units):

    .. math::

        T(\theta)     = & T_{\mathrm{user}}(\theta) \\
        V(\theta)     = & V_{\mathrm{user}}(\theta)

    where :math:`\theta` is the angle from A-B to B-C in the triple.

    :math:`T_{\mathrm{user}}(\theta)` and :math:`V_{\mathrm{user}}(\theta)` are evaluated on *width* grid points
    between :math:`0` and :math:`\pi`. Values are interpolated linearly between grid points.
    For correctness, you must specify: :math:`T = -\frac{\partial V}{\partial \theta}`

    Parameters:

    - :math:`T_{\mathrm{user}}(\theta)` and :math:`V_{\mathrm{user}}(\theta)` - evaluated by ``func`` (see example)
    - coefficients passed to ``func`` - ``angle_coeff`` (see example)

    The table *width* is set once when :py:class:`table` is specified. There are two ways to specify the other
    parameters.

    .. rubric:: Set table from a given function

    When you have a functional form for T and F, you can enter that
    directly into python. :py:class:`table` will evaluate the given function over *width* points between :math:`0` and :math:`\pi`
    and use the resulting values in the table::

        def harmonic(theta, kappa, theta_0):
            V = 0.5 * kappa * (theta-theta_0)**2;
            T = -kappa*(theta-theta_0);
            return (V, T)

        btable = angle.table(width=1000)
        btable.angle_coeff.set('angle1', func=harmonic, coeff=dict(kappa=330, theta_0=0))
        btable.angle_coeff.set('angle2', func=harmonic,coeff=dict(kappa=30, theta_0=0.1))

    .. rubric:: Set a table from a file

    When you have no function for for *T* or *F*, or you otherwise have the data listed in a file, :py:class:`table` can use the given
    values directly. You must first specify the number of rows in your tables when initializing :py:class:`table`. Then use
    :py:meth:`set_from_file()` to read the file::

        btable = angle.table(width=1000)
        btable.set_from_file('polymer', 'angle.dat')

    """
    def __init__(self, width, name=None):

        # initialize the base class
        force._force.__init__(self, name);


        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.TableAngleForceCompute(hoomd.context.current.system_definition, int(width), self.name);
        else:
            self.cpp_force = _md.TableAngleForceComputeGPU(hoomd.context.current.system_definition, int(width), self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient matrix
        self.angle_coeff = coeff();

        # stash the width for later use
        self.width = width;

    def update_angle_table(self, atype, func, coeff):
        # allocate arrays to store V and F
        Vtable = _hoomd.std_vector_scalar();
        Ttable = _hoomd.std_vector_scalar();

        # calculate dth
        dth = math.pi / float(self.width-1);

        # evaluate each point of the function
        for i in range(0, self.width):
            theta = dth * i;
            (V,T) = func(theta, **coeff);

            # fill out the tables
            Vtable.append(V);
            Ttable.append(T);

        # pass the tables on to the underlying cpp compute
        self.cpp_force.setTable(atype, Vtable, Ttable);


    def update_coeffs(self):
        # check that the angle coefficients are valid
        if not self.angle_coeff.verify(["func", "coeff"]):
            hoomd.context.current.device.cpp_msg.error("Not all angle coefficients are set for angle.table\n");
            raise RuntimeError("Error updating angle coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getAngleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getAngleData().getNameByType(i));


        # loop through all of the unique type angles and evaluate the table
        for i in range(0,ntypes):
            func = self.angle_coeff.get(type_list[i], "func");
            coeff = self.angle_coeff.get(type_list[i], "coeff");

            self.update_angle_table(i, func, coeff);

    def set_from_file(self, anglename, filename):
        R""" Set a angle pair interaction from a file.

        Args:
            anglename (str): Name of angle
            filename (str): Name of the file to read

        The provided file specifies V and F at equally spaced theta values::

            #t  V    T
            0.0 2.0 -3.0
            1.5707 3.0 -4.0
            3.1414 2.0 -3.0

        Warning:
            The theta values are not used by the code.  It is assumed that a table that has N rows will start at 0, end at :math:`\pi`
            and that :math:`\delta \theta = \pi/(N-1)`. The table is read
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
                hoomd.context.current.device.cpp_msg.error("angle.table: file must have exactly 3 columns\n");
                raise RuntimeError("Error reading table file");

            # append to the tables
            theta_table.append(values[0]);
            V_table.append(values[1]);
            T_table.append(values[2]);

        # validate input
        if self.width != len(theta_table):
            hoomd.context.current.device.cpp_msg.error("angle.table: file must have exactly " + str(self.width) + " rows\n");
            raise RuntimeError("Error reading table file");


        # check for even spacing
        dth = math.pi / float(self.width-1);
        for i in range(0,self.width):
            theta =  dth * i;
            if math.fabs(theta - theta_table[i]) > 1e-3:
                hoomd.context.current.device.cpp_msg.error("angle.table: theta must be monotonically increasing and evenly spaced\n");
                raise RuntimeError("Error reading table file");

        self.angle_coeff.set(anglename, func=_table_eval, coeff=dict(V=V_table, T=T_table, width=self.width))

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['angle_coeff'] = self.angle_coeff
        return data
