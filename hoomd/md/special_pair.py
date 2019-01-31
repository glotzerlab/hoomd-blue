# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: jglaser / All Developers are free to add commands for new features

R""" Potentials between special pairs of particles

Special pairs are used to implement interactions between designated pairs of particles.
They act much like bonds, except that the interaction potential is typically a pair potential,
such as LJ.

By themselves, special pairs that have been specified in an initial configuration do nothing. Only when you
specify an force (i.e. special_pairs.lj), are forces actually calculated between the
listed particles.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force;
from hoomd.md import bond;
import hoomd;

import math;
import sys;

class coeff:
    R""" Define special_pair coefficients.

    The coefficients for all special pair potentials are specified using this class. Coefficients are
    specified per pair type.

    There are two ways to set the coefficients for a particular special_pair potential.
    The first way is to save the special_pair potential in a variable and call :py:meth:`set()` directly.
    See below for an example of this.

    The second method is to build the coeff class first and then assign it to the
    special_pair potential. There are some advantages to this method in that you could specify a
    complicated set of special_pair potential coefficients in a separate python file and import
    it into your job script.

    Example::

        my_coeffs = hoomd.md.special_pair.coeff();
        special_pair_force.pair_coeff.set('pairtype1', epsilon=1, sigma=1)
        special_pair_force.pair_coeff.set('backbone', epsilon=1.2, sigma=1)

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
        R""" Sets parameters for special_pair types.

        Args:
            type (str): Type of special_pair (or a list of type names)
            coeffs: Named coefficients (see below for examples)

        Calling :py:meth:`set()` results in one or more parameters being set for a special_pair type. Types are identified
        by name, and parameters are also added by name. Which parameters you need to specify depends on the special_pair
        potential you are setting these coefficients for, see the corresponding documentation.

        All possible special_pair types as defined in the simulation box must be specified before executing run().
        You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
        special_pair types that do not exist in the simulation. This can be useful in defining a potential field for many
        different types of special_pairs even when some simulations only include a subset.

        Examples::

            my_special_pair_force.special_pair_coeff.set('pair1', epsilon=1, sigma=1)
            my_special_pair_force.pair_coeff.set('pair2', epsilon=0.5, sigma=0.7)
            my_special_pair_force.pair_coeff.set(['special_pairA','special_pairB'], epsilon=0, sigma=1)

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
            hoomd.context.msg.error("Cannot verify special_pair coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getPairData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getPairData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                hoomd.context.msg.error("Pair type " +str(type) + " not found in pair coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    hoomd.context.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the special pair force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                hoomd.context.msg.error("Special pair type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %special_pair %force coefficient
    # \detail
    # \param type Name of special_pair type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            hoomd.context.msg.error("Bug detected in force.coeff. Please report\n");
            raise RuntimeError("Error setting special_pair coeff");

        return self.values[type][coeff_name];

    ## \internal
    # \brief Return metadata
    def get_metadata(self):
        return self.values


## \internal
# \brief Base class for special pair potentials
#
# A special pair in hoomd.* reflects a PotentialSpecialPair in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd
# writers. 1) The instance of the c++ bond force itself is tracked and added to the
# System 2) methods are provided for disabling the force from being added to the
# net force on each particle
class _special_pair(force._force):
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

        # setup the coefficient vector (use bond coefficients for that)
        self.pair_coeff = coeff();

        self.enabled = True;

    def update_coeffs(self):
        coeff_list = self.required_coeffs;
        # check that the force coefficients are valid
        if not self.pair_coeff.verify(coeff_list):
           hoomd.context.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getPairData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getPairData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.pair_coeff.get(type_list[i], name);

            param = self.process_coeff(coeff_dict);
            self.cpp_force.setParams(i, param);

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['pair_coeff'] = self.pair_coeff
        return data

class lj(_special_pair):
    R""" LJ special pair potential.

    Args:
        name (str): Name of the special_pair instance.

    :py:class:`lj` specifies a Lennard-Jones potential energy between the two particles in each defined pair.

    This is useful for implementing e.g. special 1-4 interactions in all-atom force fields.

    The pair potential uses the standard LJ definition.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{LJ}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
                          \alpha \left( \frac{\sigma}{r} \right)^{6} \right] & r < r_{\mathrm{cut}} \\
                            = & 0 & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    where :math:`\vec{r}` is the vector pointing from one particle to the other in the bond.

    Coefficients:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)

    Example::

        lj = special_pair.lj(name="my_pair")
        lj.pair_coeff.set('pairtype_1', epsilon=5.4, sigma=0.47, r_cut=1.1)

    Note:
        The energy of special pair interactions is reported in a log quantity **special_pair_lj_energy**, which
        is separate from those of other non-bonded interactions. Therefore, the total energy of nonbonded interactions
        is obtained by adding that of standard and special interactions.

    .. versionadded:: 2.1

    """
    def __init__(self,name=None):
        hoomd.util.print_status_line();

        # initialize the base class
        _special_pair.__init__(self);

        # check that some bonds are defined
        if hoomd.context.current.system_definition.getPairData().getNGlobal() == 0:
            hoomd.context.msg.error("No pairs are defined.\n");
            raise RuntimeError("Error creating special pair forces");

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.PotentialSpecialPairLJ(hoomd.context.current.system_definition,self.name);
        else:
            self.cpp_force = _md.PotentialSpecialPairLJGPU(hoomd.context.current.system_definition,self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = ['epsilon','sigma','alpha','r_cut'];
        self.pair_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        r_cut = coeff['r_cut'];
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        r_cut_squared = r_cut * r_cut
        return _hoomd.make_scalar3(lj1, lj2, r_cut_squared);


class coulomb(_special_pair):
    R""" Coulomb special pair potential.

    Args:
        name (str): Name of the special_pair instance.

    :py:class:`coulomb` specifies a Coulomb potential energy between the two particles in each defined pair.

    This is useful for implementing e.g. special 1-4 interactions in all-atom force fields. It uses a standard Coulomb interaction with a scaling parameter. This allows for using this for scaled 1-4 interactions like in OPLS where both the 1-4 LJ and Coulomb interactions are scaled by 0.5.

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{Coulomb}}(r)  = & \alpha \cdot \left[ \frac{q_{a}q_{b}}{r} \right] & r < r_{\mathrm{cut}} \\
                            = & 0 & r \ge r_{\mathrm{cut}} \\
        \end{eqnarray*}

    where :math:`\vec{r}` is the vector pointing from one particle to the other in the bond.

    Coefficients:

    - :math:`\alpha` - Coulomb scaling factor (defaults to 1.0)
    - :math:`q_{a}` - charge of particle a (in hoomd charge units)
    - :math:`q_{b}` - charge of particle b (in hoomd charge units)
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)

    Example::

        coul = special_pair.coulomb(name="myOPLS_style")
        coul.pair_coeff.set('pairtype_1', alpha=0.5, r_cut=1.1)

    Note:
        The energy of special pair interactions is reported in a log quantity **special_pair_coul_energy**, which
        is separate from those of other non-bonded interactions. Therefore, the total energy of non-bonded interactions
        is obtained by adding that of standard and special interactions.

    .. versionadded:: 2.2
    .. versionchanged:: 2.2

    """
    def __init__(self, name=None):
        hoomd.util.print_status_line();

        # initialize the base class
        _special_pair.__init__(self);

        # check that some bonds are defined
        if hoomd.context.current.system_definition.getPairData().getNGlobal() == 0:
            hoomd.context.msg.error("No pairs are defined.\n");
            raise RuntimeError("Error creating special pair forces");

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.PotentialSpecialPairCoulomb(hoomd.context.current.system_definition,self.name);
        else:
            self.cpp_force = _md.PotentialSpecialPairCoulombGPU(hoomd.context.current.system_definition,self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = ['alpha', 'r_cut'];
        self.pair_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        r_cut = coeff['r_cut'];
        alpha = coeff['alpha'];

        r_cut_squared = r_cut * r_cut;
        return _hoomd.make_scalar2(alpha, r_cut_squared);

