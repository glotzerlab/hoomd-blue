# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" External forces.

Apply an external force to all particles in the simulation. This module organizes all external forces.
As an example, a force derived from a :py:class:`periodic` potential can be used to induce a concentration modulation
in the system.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force;
import hoomd;
from hoomd import meta

import sys;
import math;

class coeff(force._coeff):
    R""" Defines external potential coefficients.

    The coefficients for all external forces are specified using this class. Coefficients are specified per particle
    type.

    Example::

        my_external_force.force_coeff.set('A', A=1.0, i=1, w=0.02, p=3)
        my_external_force.force_coeff.set('B', A=-1.0, i=1, w=0.02, p=3)

    """

    def set(self, type, **coeffs):
        R""" Sets parameters for particle types.

        Args:
            type (str): Type of particle (or list of types)
            coeff: Named coefficients (see below for examples)

        Calling :py:meth:`set()` results in one or more parameters being set for a particle type. Types are identified
        by name, and parameters are also added by name. Which parameters you need to specify depends on the external
        force you are setting these coefficients for, see the corresponding documentation.

        All possible particle types as defined in the simulation box must be specified before executing run().
        You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
        particle types that do not exist in the simulation. This can be useful in defining a force field for many
        different types of particles even when some simulations only include a subset.

        To set the same coefficients between many particle types, provide a list of type names instead of a single
        one. All types in the list will be set to the same parameters. A convenient wildcard that lists all types
        of particles in the simulation can be gotten from a saved system from the init command.

        Examples::

            coeff.set('A', A=1.0, i=1, w=0.02, p=3)
            coeff.set('B', A=-1.0, i=1, w=0.02, p=3)
            coeff.set(['A','B'], i=1, w=0.02, p=3)

        Note:
            Single parameters can be updated. For example,
            executing ``coeff.set('A', A=1.0)`` will update the value of ``A`` and leave the other parameters as they
            were previously set.

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
            hoomd.context.msg.error("Cannot verify force coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys() and len(required_coeffs):
                hoomd.context.msg.error("Particle type " + type + " is missing required coefficients\n");
                return False

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    hoomd.context.msg.notice(3, "Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) +\
                          ", but is not used by the external force");
                else:
                    count += 1;

            if count != len(required_coeffs):
                hoomd.context.msg.error("Particle type " + type + " is missing required coefficients\n");
                valid = False;
        return valid;

    ## \internal
    # \brief Gets the value of a single external force coefficient
    # \detail
    # \param type Name of particle type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values:
            hoomd.context.msg.error("Bug detected in external.coeff. Please report\n");
            raise RuntimeError("Error setting external coeff");

        return self.values[type][coeff_name];

## \internal
# \brief Base class for external forces
#
# An external_force in hoomd reflects a PotentialExternal in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd
# writers. 1) The instance of the c++ external force itself is tracked and added to the
# System 2) methods are provided for disabling the force from being added to the
# net force on each particle
class _external_force(force._force):
    ## \internal
    # \brief Constructs the external force
    #
    # \param name name of the external force instance
    #
    # Initializes the cpp_force to None.
    # If specified, assigns a name to the instance
    # Assigns a name to the force in force_name;
    def __init__(self, name=""):
        # initialize the base class
        force._force.__init__(self, name);
        self.metadata_fields += ['force_coeff', 'field_coeff']

        self.cpp_force = None;

        # setup the coefficient vector
        self.force_coeff = coeff();
        self.field_coeff = None;

        self.name = name
        self.enabled = True;

        # create force data iterator
        self.external_forces = hoomd.data.force_data(self);

    def update_coeffs(self):
        coeff_list = self.required_coeffs;

        if self.field_coeff:
            fcoeff = self.process_field_coeff(self.field_coeff);
            self.cpp_force.setField(fcoeff);

        if self.required_coeffs is not None:
            # check that the force coefficients are valid
            if not self.force_coeff.verify(coeff_list):
               hoomd.context.msg.error("Not all force coefficients are set\n");
               raise RuntimeError("Error updating force coefficients");

            # set all the params
            ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
            type_list = [];
            for i in range(0,ntypes):
                type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

            for i in range(0,ntypes):
                # build a dict of the coeffs to pass to proces_coeff
                coeff_dict = {};
                for name in coeff_list:
                    coeff_dict[name] = self.force_coeff.get(type_list[i], name);

                param = self.process_coeff(coeff_dict);
                self.cpp_force.setParams(i, param);

    ## \internal
    # \brief Return the metadata
    # Override parent to correctly access coefficient attribute.
    def get_metadata(self):
        # Calling two steps up the hierarchy this way is BAD DESIGN, but is
        # necessary in this case because of the heavy inconsistency in the way
        # the *_coeff class attributes are names. A cleaner fix would be to
        # standardize the name across all subclasses of force._force (with
        # aliases for backwards compatibility, but for now calling up to higher
        # in the class hierarchy manually will have to do.
        metadata = hoomd.meta._metadata.get_metadata(self)

        # The only reason the parent method is overriden is to manually pop the
        # correct key from the dictionary.
        coeff_obj = metadata[meta.META_KEY_TRACKED].pop('force_coeff')
        parameters = {}
        for k, v in coeff_obj.values.items():
            parameters[k] = v
        metadata[meta.META_KEY_TRACKED]['parameters'] = parameters
        return metadata

    # Override parent method to use the right coeff object name.
    @classmethod
    def from_metadata(cls, params, args=[]):
        # Note that we're calling super on force._force to bypass its method.
        obj = super(force._force, cls).from_metadata(params)
        for type_name, parameters in params[meta.META_KEY_TRACKED]['parameters'].items():
            obj.force_coeff.set(type_name, **parameters)
        return obj

class periodic(_external_force):
    R""" One-dimension periodic potential.

    :py:class:`periodic` specifies that an external force should be
    added to every particle in the simulation to induce a periodic modulation
    in the particle concentration. The force parameters can be set on a per particle
    type basis. The potential can e.g. be used to induce an ordered phase in a block-copolymer melt.

    The external potential :math:`V(\vec{r})` is implemented using the following formula:

    .. math::

       V(\vec{r}) = A * \tanh\left[\frac{1}{2 \pi p w} \cos\left(p \vec{b}_i\cdot\vec{r}\right)\right]

    where :math:`A` is the ordering parameter, :math:`\vec{b}_i` is the reciprocal lattice vector direction
    :math:`i=0..2`, :math:`p` the periodicity and :math:`w` the interface width
    (relative to the distance :math:`2\pi/|\mathbf{b_i}|` between planes in the :math:`i`-direction).
    The modulation is one-dimensional. It extends along the lattice vector :math:`\mathbf{a}_i` of the
    simulation cell.

    Examples::

        # Apply a periodic composition modulation along the first lattice vector
        periodic = external.periodic()
        periodic.force_coeff.set('A', A=1.0, i=0, w=0.02, p=3)
        periodic.force_coeff.set('B', A=-1.0, i=0, w=0.02, p=3)
    """
    def __init__(self, name=""):
        hoomd.util.print_status_line();

        # initialize the base class
        _external_force.__init__(self, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.PotentialExternalPeriodic(hoomd.context.current.system_definition,self.name);
        else:
            self.cpp_force = _md.PotentialExternalPeriodicGPU(hoomd.context.current.system_definition,self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = ['A','i','w','p'];

    def process_coeff(self, coeff):
        A = coeff['A'];
        i = coeff['i'];
        w = coeff['w'];
        p = coeff['p'];

        return _hoomd.make_scalar4(_hoomd.int_as_scalar(i), A, w, _hoomd.int_as_scalar(p));

class e_field(_external_force):
    R""" Electric field.

    :py:class:`e_field` specifies that an external force should be
    added to every particle in the simulation that results from an electric field.

    The external potential :math:`V(\vec{r})` is implemented using the following formula:

    .. math::

       V(\vec{r}) = - q_i \vec{E} \cdot \vec{r}


    where :math:`q_i` is the particle charge and :math:`\vec{E}` is the field vector

    Example::

        # Apply an electric field in the x-direction
        e_field = external.e_field((1,0,0))
    """
    def __init__(self, field, name=""):
        hoomd.util.print_status_line();

        # initialize the base class
        _external_force.__init__(self, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.PotentialExternalElectricField(hoomd.context.current.system_definition,self.name);
        else:
            self.cpp_force = _md.PotentialExternalElectricFieldGPU(hoomd.context.current.system_definition,self.name);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = None;

        self.field_coeff = tuple(field)

    def process_coeff(self, coeff):
        pass;

    def process_field_coeff(self, field):
        return _hoomd.make_scalar3(field[0],field[1],field[2])
