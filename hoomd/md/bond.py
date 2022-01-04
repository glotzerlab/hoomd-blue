# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Bond potentials."""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.md import force
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
import hoomd

import math


class Bond(Force):
    """Constructs the bond potential.

    Note:
        :py:class:`Bond` is the base class for all bond potentials.
        Users should not instantiate this class directly.
    """

    def _attach(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        # TODO remove string argument
        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)

        super()._attach()


class Harmonic(Bond):
    r"""Harmonic bond potential.

    :py:class:`Harmonic` specifies a harmonic potential energy between the two
    particles in each defined bond.

    .. math::

        V(r) = \frac{1}{2} k \left( r - r_0 \right)^2

    where :math:`r` is the distance from one particle to the other in the bond.

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the harmonic bonds for each particle type.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - potential constant
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``r0`` (`float`, **required**) - rest length
              :math:`[\mathrm{length}]`

    Examples::

        harmonic = bond.Harmonic()
        harmonic.params['polymer'] = dict(k=3.0, r0=2.38)
        harmonic.params['backbone'] = dict(k=10.0, r0=1.0)
    """
    _cpp_class_name = "PotentialBondHarmonic"

    def __init__(self):
        params = TypeParameter("params", "bond_types",
                               TypeParameterDict(k=float, r0=float, len_keys=1))
        self._add_typeparam(params)


class FENE(Bond):
    r"""FENE bond potential.

    :py:class:`FENE` specifies a FENE potential energy between the two particles
    in each defined bond.

    .. math::

        V(r) = - \frac{1}{2} k r_0^2 \ln \left( 1 - \left( \frac{r -
               \Delta}{r_0} \right)^2 \right) + V_{\mathrm{WCA}}(r)

    where :math:`r` is the distance from one particle to the other in the bond,
    :math:`\Delta = (d_i + d_j)/2 - 1`, :math:`d_i` is the diameter of particle
    :math:`i`, and

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{WCA}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r -
                                 \Delta} \right)^{12} - \left( \frac{\sigma}{r -
                                 \Delta} \right)^{6} \right]  + \varepsilon;
                               & r-\Delta < 2^{\frac{1}{6}}\sigma\\
                             = & 0;
                               & r-\Delta \ge 2^{\frac{1}{6}}\sigma
        \end{eqnarray*}

    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the FENE potential bonds.
            The dictionary has the following keys:

            * ``k`` (`float`, **required**) - attractive force strength
              :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`

            * ``r0`` (`float`, **required**) - size parameter
              :math:`[\mathrm{length}]`

            * ``epsilon`` (`float`, **required**) - repulsive force strength
              :math:`[\mathrm{energy}]`

            * ``sigma`` (`float`, **required**) - repulsive force interaction
              :math:`[\mathrm{length}]`

    Examples::

        bond_potential = bond.FENE()
        bond_potential.params['molecule'] = dict(k=3.0, r0=2.38,
                                                 epsilon=1.0, sigma=1.0)
        bond_potential.params['backbone'] = dict(k=10.0, r0=1.0,
                                                 epsilon=0.8, sigma=1.2)

    """
    _cpp_class_name = "PotentialBondFENE"

    def __init__(self):
        params = TypeParameter(
            "params", "bond_types",
            TypeParameterDict(k=float,
                              r0=float,
                              epsilon=float,
                              sigma=float,
                              len_keys=1))
        self._add_typeparam(params)


def _table_eval(r, rmin, rmax, V, F, width):
    dr = (rmax - rmin) / float(width - 1)
    i = int(round((r - rmin) / dr))
    return (V[i], F[i])


class table(force._force):  # noqa - Will be renamed when updated for v3
    r"""Tabulated bond potential.

    Args:
        width (int): Number of points to use to interpolate V and F
        name (str): Name of the potential instance

    :py:class:`table` specifies that a tabulated bond potential should be
    applied between the two particles in each defined bond.

    The force :math:`\vec{F}` is (in force units) and the potential :math:`V(r)`
    is (in energy units):

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}(\vec{r})     = & 0;
                               & r < r_{\mathrm{min}} \\
                             = & F_{\mathrm{user}}(r)\hat{r};
                               & r < r_{\mathrm{max}} \\
                             = & 0;
                               & r \ge r_{\mathrm{max}} \\
                             \\
        V(r)       = & 0                    & r < r_{\mathrm{min}} \\
                   = & V_{\mathrm{user}}(r) & r < r_{\mathrm{max}} \\
                   = & 0                    & r \ge r_{\mathrm{max}} \\
        \end{eqnarray*}

    where :math:`r` is the distance from one particle to the other in the bond.
    Care should be taken to define the range of the bond so that it is not
    possible for the distance between two bonded particles to be outside the
    specified range. On the CPU, this will throw an error. On the GPU, this
    will throw an error if GPU error checking is enabled.

    :math:`F_{\mathrm{user}}(r)` and :math:`V_{\mathrm{user}}(r)` are evaluated
    on *width* grid points between :math:`r_{\mathrm{min}}` and
    :math:`r_{\mathrm{max}}`. Values are interpolated linearly between grid
    points. For correctness, you must specify the force defined by: :math:`F =
    -\frac{\partial V}{\partial r}`

    The following coefficients must be set for each bond type:

    - :math:`F_{\mathrm{user}}(r)` and :math:`V_{\mathrm{user}}(r)` - evaluated
      by ``func`` (see example)
    - coefficients passed to ``func`` - ``coeff`` (see example)
    - :math:`r_{\mathrm{min}}` - ``rmin`` :math:`[\mathrm{length}]`
    - :math:`r_{\mathrm{max}}` - ``rmax`` :math:`[\mathrm{length}]`

    The table *width* is set once when bond.table is specified.
    There are two ways to specify the other parameters.

    .. rubric:: Set table from a given function

    When you have a functional form for V and F, you can enter that directly
    into python. :py:class:`table` will evaluate the given function over *width*
    points between *rmin* and *rmax* and use the resulting values in the table::

        def harmonic(r, rmin, rmax, kappa, r0):
           V = 0.5 * kappa * (r-r0)**2;
           F = -kappa*(r-r0);
           return (V, F)

        btable = bond.table(width=1000)
        btable.bond_coeff.set('bond1', func=harmonic, rmin=0.2, rmax=5.0,
                              coeff=dict(kappa=330, r0=0.84))
        btable.bond_coeff.set('bond2', func=harmonic, rmin=0.2, rmax=5.0,
                              coeff=dict(kappa=30, r0=1.0))

    .. rubric:: Set a table from a file

    When you have no function for for *V* or *F*, or you otherwise have the data
    listed in a file, :py:class:`table` can use the given values directly. You
    must first specify the number of rows in your tables when initializing
    bond.table. Then use :py:meth:`set_from_file()` to read the file::

        btable = bond.table(width=1000)
        btable.set_from_file('polymer', 'btable.file')

    Note:
        For potentials that diverge near r=0, make sure to set ``rmin`` to a
        reasonable value. If a potential does
        not diverge near r=0, then a setting of ``rmin=0`` is valid.

    Note:
        Ensure that ``rmin`` and ``rmax`` cover the range of possible bond
        lengths. When gpu error checking is on, a error will
        be thrown if a bond distance is outside than this range.
    """

    def __init__(self, width, name=None):

        # initialize the base class
        force._force.__init__(self, name)

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.BondTablePotential(
                hoomd.context.current.system_definition, int(width), self.name)
        else:
            self.cpp_force = _md.BondTablePotentialGPU(
                hoomd.context.current.system_definition, int(width), self.name)

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

        # setup the coefficients matrix
        self.bond_coeff = coeff()  # noqa - this will be rewritten for v3

        # stash the width for later use
        self.width = width

    def update_bond_table(self, btype, func, rmin, rmax, coeff):  # noqa
        # allocate arrays to store V and F
        Vtable = _hoomd.std_vector_scalar()
        Ftable = _hoomd.std_vector_scalar()

        # calculate dr
        dr = (rmax - rmin) / float(self.width - 1)

        # evaluate each point of the function
        for i in range(0, self.width):
            r = rmin + dr * i
            (V, F) = func(r, rmin, rmax, **coeff)

            # fill out the tables
            Vtable.append(V)
            Ftable.append(F)

        # pass the tables on to the underlying cpp compute
        self.cpp_force.setTable(btype, Vtable, Ftable, rmin, rmax)

    def update_coeffs(self):  # noqa
        # check that the bond coefficients are valid
        if not self.bond_coeff.verify(["func", "rmin", "rmax", "coeff"]):
            hoomd.context.current.device.cpp_msg.error(
                "Not all bond coefficients are set for bond.table\n")
            raise RuntimeError("Error updating bond coefficients")

        # set all the params
        ntypes = hoomd.context.current.system_definition.getBondData(
        ).getNTypes()
        type_list = []
        for i in range(0, ntypes):
            type_list.append(hoomd.context.current.system_definition
                             .getBondData().getNameByType(i))

        # loop through all of the unique type bonds and evaluate the table
        for i in range(0, ntypes):
            func = self.bond_coeff.get(type_list[i], "func")
            rmin = self.bond_coeff.get(type_list[i], "rmin")
            rmax = self.bond_coeff.get(type_list[i], "rmax")
            coeff = self.bond_coeff.get(type_list[i], "coeff")

            self.update_bond_table(i, func, rmin, rmax, coeff)

    def set_from_file(self, bondname, filename):  # noqa
        r"""Set a bond pair interaction from a file.

        Args: bondname (str): Name of bond filename (str): Name of the file to
            read

        The provided file specifies V and F at equally spaced r values.
        Example::

            #r  V    F
            1.0 2.0 -3.0
            1.1 3.0 -4.0
            1.2 2.0 -3.0
            1.3 1.0 -2.0
            1.4 0.0 -1.0
            1.5 -1.0 0.0

        The first r value sets ``rmin``, the last sets ``rmax``. Any line with
        # as the first non-whitespace character is
        is treated as a comment. The ``r`` values must monotonically increase
        and be equally spaced. The table is read directly into the grid points
        used to evaluate :math:`F_{\mathrm{user}}(r)` and
        :math:`V_{\mathrm{user}}(r)`.
        """
        # open the file
        f = open(filename)

        r_table = []
        V_table = []
        F_table = []

        # read in lines from the file
        for line in f.readlines():
            line = line.strip()

            # skip comment lines
            if line[0] == '#':
                continue

            # split out the columns
            cols = line.split()
            values = [float(f) for f in cols]

            # validate the input
            if len(values) != 3:
                hoomd.context.current.device.cpp_msg.error(
                    "bond.table: file must have exactly 3 columns\n")
                raise RuntimeError("Error reading table file")

            # append to the tables
            r_table.append(values[0])
            V_table.append(values[1])
            F_table.append(values[2])

        # validate input
        if self.width != len(r_table):
            hoomd.context.current.device.cpp_msg.error(
                "bond.table: file must have exactly " + str(self.width)
                + " rows\n")
            raise RuntimeError("Error reading table file")

        # extract rmin and rmax
        rmin_table = r_table[0]
        rmax_table = r_table[-1]

        # check for even spacing
        dr = (rmax_table - rmin_table) / float(self.width - 1)
        for i in range(0, self.width):
            r = rmin_table + dr * i
            if math.fabs(r - r_table[i]) > 1e-3:
                hoomd.context.current.device.cpp_msg.error(
                    "bond.table: r must be monotonically increasing and evenly "
                    "spaced\n")
                raise RuntimeError("Error reading table file")

        self.bond_coeff.set(bondname,
                            func=_table_eval,
                            rmin=rmin_table,
                            rmax=rmax_table,
                            coeff=dict(V=V_table, F=F_table, width=self.width))


class Tether(Bond):
    r"""Tethering bond potential.

    :py:class:`Tether` specifies a Tethering potential energy between two
    particles in each defined bond.

    The tethered network is described in Refs. `Gompper, G. & Kroll, D. M.
    Statistical Mechanics of Membranes and Surfaces 2nd edn (eds Nelson, D. R.
    et al.) 359-426 (World Scientific, 2004)
    <https://www.worldscientific.com/worldscibooks/10.1142/5473>`__ and
    `Noguchi, H. & Gompper, G., Phys. Rev. E 72 011901 (2005)
    <https://link.aps.org/doi/10.1103/PhysRevE.72.011901>`__.

    .. math::

        V(r) = V_{\mathrm{att}}(r) + V_{\mathrm{rep}}(r)

    where :math:`r` is the distance from one particle to the other in the bond.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{att}}(r)  = & k_b \frac{exp(1/(l_{c0}-r)}{l_{max}-r};
                                & r > l_{c0}\\
                                = & 0;
                                & r \leq l_{c0}
        \end{eqnarray*}

    .. math::

        \begin{eqnarray*}
        V_{\mathrm{rep}}(r)  = & k_b \frac{exp(1/(r-l_{c1})}{r-l_{min}};
                                & r < l_{c1}\\
                                = & 0;
                                & r \ge l_{c1}
        \end{eqnarray*}

    .. math::
        l_{min} < l_{c1} < l_{c0} < l_{max}


    Attributes:
        params (TypeParameter[``bond type``, dict]):
            The parameter of the Tethering potential bonds.
            The dictionary has the following keys:

            * ``k_b`` (`float`, **required**) - bond stiffness
              :math:`[\mathrm{energy}]`

            * ``l_min`` (`float`, **required**) - minimum bond length
              :math:`[\mathrm{length}]`

            * ``l_c1`` (`float`, **required**) - cutoff distance of repulsive
              part :math:`[\mathrm{length}]`

            * ``l_c0`` (`float`, **required**) - cutoff distance of attractive
              part :math:`[\mathrm{length}]`

            * ``l_max`` (`float`, **required**) - maximum bond length
              :math:`[\mathrm{length}]`

    Examples::

        bond_potential = bond.Tether()
        bond_potential.params['tether'] = dict(k_b=10.0, l_min=0.9, l_c1=1.2,
                                               l_c0=1.8, l_max=2.1)
    """
    _cpp_class_name = "PotentialBondTether"

    def __init__(self):
        params = TypeParameter(
            "params", "bond_types",
            TypeParameterDict(k_b=float,
                              l_min=float,
                              l_c1=float,
                              l_c0=float,
                              l_max=float,
                              len_keys=1))
        self._add_typeparam(params)
