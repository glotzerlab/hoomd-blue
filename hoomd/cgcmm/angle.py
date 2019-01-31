# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" CGCMM angle potentials.
"""

from hoomd.md import _md
from hoomd.md import force
import hoomd
from hoomd.cgcmm import _cgcmm

import math;
import sys;

class cgcmm(force._force):
    R""" CGCMM angle potential.

    The command angle.cgcmm defines a regular harmonic potential energy between every defined triplet
    of particles in the simulation, but in addition in adds the repulsive part of a CGCMM pair potential
    between the first and the third particle.

    `B. Levine et. al. 2011 <http://dx.doi.org/10.1021/ct2005193>`_ describes the CGCMM implementation details in
    HOOMD-blue. Cite it if you utilize the CGCMM potential in your work.

    The total potential is thus:

    .. math::

        V(\theta) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2

    where :math:`\theta` is the current angle between the three particles
    and either:

    .. math::

        V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~} V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[
        \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right]
        \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot 2^{\frac{1}{6}}


    .. math::

        V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~}
        V_{\mathrm{LJ}}(r) = \frac{27}{4} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{9} -
        \left( \frac{\sigma}{r} \right)^{6} \right]
        \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot \left(\frac{3}{2}\right)^{\frac{1}{3}}

    .. math::

        V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~}
        V_{\mathrm{LJ}}(r) = \frac{3\sqrt{3}}{2} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
        \left( \frac{\sigma}{r} \right)^{4} \right]
        \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot 3^{\frac{1}{8}}

    with :math:`r_{13}` being the distance between the two outer particles of the angle.

    Coefficients:

    - :math:`\theta_0` - rest angle ``t0`` (in radians)
    - :math:`k` - potential constant ``k`` (in units of energy/radians^2)
    - :math:`\varepsilon` - strength of potential ``epsilon`` (in energy units)
    - :math:`\sigma` - distance of interaction ``sigma`` (in distance units)

    Coefficients :math:`k, \theta_0, \varepsilon``, and :math:`\sigma` and Lennard-Jones exponents pair must be set for
    each type of angle in the simulation using :py:meth:`set_coeff()`.
    """
    def __init__(self):
        hoomd.util.print_status_line();
        # check that some angles are defined
        if hoomd.context.current.system_definition.getAngleData().getNGlobal() == 0:
            hoomd.context.msg.error("No angles are defined.\n");
            raise RuntimeError("Error creating CGCMM angle forces");

        # initialize the base class
        force._force.__init__(self);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _cgcmm.CGCMMAngleForceCompute(hoomd.context.current.system_definition);
        else:
            self.cpp_force = _cgcmm.CGCMMAngleForceComputeGPU(hoomd.context.current.system_definition);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # variable for tracking which angle type coefficients have been set
        self.angle_types_set = [];

    def set_coeff(self, angle_type, k, t0, exponents, epsilon, sigma):
        R""" Sets the CG-CMM angle coefficients for a particular angle type.

        Args:
            angle_type (str): Angle type to set coefficients for
            k (float): Coefficient :math:`k` (in units of energy/radians^2)
            t0 (float): Coefficient :math:`\theta_0` (in radians)
            exponents (str): is the type of CG-angle exponents we want to use for the repulsion.
            epsilon (float): is the 1-3 repulsion strength (in energy units)
            sigma (float): is the CG particle radius (in distance units)

        Examples::

            cgcmm.set_coeff('polymer', k=3.0, t0=0.7851, exponents=126, epsilon=1.0, sigma=0.53)
            cgcmm.set_coeff('backbone', k=100.0, t0=1.0, exponents=96, epsilon=23.0, sigma=0.1)
            cgcmm.set_coeff('residue', k=100.0, t0=1.0, exponents='lj12_4', epsilon=33.0, sigma=0.02)
            cgcmm.set_coeff('cg96', k=100.0, t0=1.0, exponents='LJ9-6', epsilon=9.0, sigma=0.3)

        """
        hoomd.util.print_status_line();
        cg_type=0

        # set the parameters for the appropriate type
        if (exponents == 124) or  (exponents == 'lj12_4') or  (exponents == 'LJ12-4') :
            cg_type=2;

            self.cpp_force.setParams(hoomd.context.current.system_definition.getAngleData().getTypeByName(angle_type),
                                     k,
                                     t0,
                                     cg_type,
                                     epsilon,
                                     sigma);

        elif (exponents == 96) or  (exponents == 'lj9_6') or  (exponents == 'LJ9-6') :
            cg_type=1;

            self.cpp_force.setParams(hoomd.context.current.system_definition.getAngleData().getTypeByName(angle_type),
                                     k,
                                     t0,
                                     cg_type,
                                     epsilon,
                                     sigma);

        elif (exponents == 126) or  (exponents == 'lj12_6') or  (exponents == 'LJ12-6') :
            cg_type=3;

            self.cpp_force.setParams(hoomd.context.current.system_definition.getAngleData().getTypeByName(angle_type),
                                     k,
                                     t0,
                                     cg_type,
                                     epsilon,
                                     sigma);
        else:
            raise RuntimeError("Unknown exponent type.  Must be 'none' or one of MN, ljM_N, LJM-N with M/N in 12/4, 9/6, or 12/6");

        # track which particle types we have set
        if not angle_type in self.angle_types_set:
            self.angle_types_set.append(angle_type);

    def update_coeffs(self):
        # get a list of all angle types in the simulation
        ntypes = hoomd.context.current.system_definition.getAngleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getAngleData().getNameByType(i));

        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.angle_types_set:
                hoomd.context.msg.error(str(cur_type) + " coefficients missing in angle.cgcmm\n");
                raise RuntimeError("Error updating coefficients");
