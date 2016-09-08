# Copyright (c) 2009-2016 The Regents of the University of Michigan
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

## \internal
# \brief Base class for special pair potentials
#
# A special pair in hoomd.* reflects a PotentialSpecialPair in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
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
        self.pair_coeff = bond.coeff();

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

        lj = special_pair.harmonic(name="mybond")
        lj.pair_coeff.set('pairtype_1', epsilon=5.4, sigma=0.47, r_cut=1.1)

    .. versionadded:: 2.1

    """
    def __init__(self,name=None):
        hoomd.util.print_status_line();

        # initiailize the base class
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
        return _hoomd.make_scalar3(lj1, lj2,r_cut);


