# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Improper potentials.

Impropers add forces between specified quadruplets of particles and are typically used to
model rotation about chemical bonds without having bonds to connect the atoms. Their most
common use is to keep structural elements flat, i.e. model the effect of conjugated
double bonds, like in benzene rings and its derivatives.

By themselves, impropers that have been specified in an input file do nothing. Only when you
specify an improper force (i.e. improper.harmonic), are forces actually calculated between the
listed particles.
"""

from hoomd.md import _md
from hoomd.md import force;
import hoomd;

import math;
import sys;

class coeff:
    R""" Define improper coefficients.

    The coefficients for all improper force are specified using this class. Coefficients are
    specified per improper type.

    Examples::

        my_coeffs = improper.coeff();
        my_improper_force.improper_coeff.set('polymer', k=330.0, r=0.84)
        my_improper_force.improper_coeff.set('backbone', k=330.0, r=0.84)
    """
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
        R""" Sets parameters for one improper type.

        Args:
            type (str): Type of improper (or list of types).
            coeffs: Named coefficients (see below for examples)

        Calling :py:meth:`set()` results in one or more parameters being set for a improper type. Types are identified
        by name, and parameters are also added by name. Which parameters you need to specify depends on the improper
        force you are setting these coefficients for, see the corresponding documentation.

        All possible improper types as defined in the simulation box must be specified before executing :py:mod:`hoomd.run()`.
        You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
        improper types that do not exist in the simulation. This can be useful in defining a force field for many
        different types of impropers even when some simulations only include a subset.

        To set the same coefficients between many particle types, provide a list of type names instead of a single
        one. All types in the list will be set to the same parameters.

        Examples::

            my_improper_force.improper_coeff.set('polymer', k=330.0, r0=0.84)
            my_improper_force.improper_coeff.set('backbone', k=1000.0, r0=1.0)
            my_improper_force.improper_coeff.set(['improperA','improperB'], k=100, r0=0.0)

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
            hoomd.context.msg.error("Cannot verify improper coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getImproperData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getImproperData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                hoomd.context.msg.error("Improper type " +str(type) + " not found in improper coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    hoomd.context.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the improper force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                hoomd.context.msg.error("Improper type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %improper %force coefficient
    # \detail
    # \param type Name of improper type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            hoomd.context.msg.error("Bug detected in force.coeff. Please report\n");
            raise RuntimeError("Error setting improper coeff");

        return self.values[type][coeff_name];

    ## \internal
    # \brief Return metadata
    def get_metadata(self):
        return self.values

class harmonic(force._force):
    R""" Harmonic improper potential.

    The command improper.harmonic specifies a harmonic improper potential energy between every quadruplet of particles
    in the simulation.

    .. math::

        V(r) = \frac{1}{2}k \left( \chi - \chi_{0}  \right )^2

    where :math:`\chi` is angle between two sides of the improper.

    Coefficients:

    - :math:`k` - strength of force, ``k`` (in energy units)
    - :math:`\chi_{0}` - equilibrium angle, ``chi`` (in radians)

    Coefficients :math:`k` and :math:`\chi_0` must be set for each type of improper in the simulation using
    :py:meth:`improper_coeff.set() <coeff.set()>`.

    Examples::

        harmonic.improper_coeff.set('heme-ang', k=30.0, chi=1.57)
        harmonic.improper_coeff.set('hydro-bond', k=20.0, chi=1.57)

    """
    def __init__(self):
        hoomd.util.print_status_line();
        # check that some impropers are defined
        if hoomd.context.current.system_definition.getImproperData().getNGlobal() == 0:
            hoomd.context.msg.error("No impropers are defined.\n");
            raise RuntimeError("Error creating improper forces");

        # initialize the base class
        force._force.__init__(self);

        self.improper_coeff = coeff();

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.HarmonicImproperForceCompute(hoomd.context.current.system_definition);
        else:
            self.cpp_force = _md.HarmonicImproperForceComputeGPU(hoomd.context.current.system_definition);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        self.required_coeffs = ['k', 'chi'];

    ## \internal
    # \brief Update coefficients in C++
    def update_coeffs(self):
        coeff_list = self.required_coeffs;
        # check that the force coefficients are valid
        if not self.improper_coeff.verify(coeff_list):
           hoomd.context.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getImproperData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getImproperData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.improper_coeff.get(type_list[i], name);

            self.cpp_force.setParams(i, coeff_dict['k'], coeff_dict['chi']);

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['improper_coeff'] = self.improper_coeff
        return data
