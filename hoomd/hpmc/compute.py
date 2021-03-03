# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Compute properties of hard particle configurations.
"""

from __future__ import print_function

from hoomd import _hoomd
from hoomd.operation import Compute
from hoomd.hpmc import _hpmc
from hoomd.hpmc import integrate
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.logging import log
import hoomd

class FreeVolume(Compute):
    """Compute the free volume available to a test particle by stochastic integration.

    Args:
        test_particle_type (str): Type of particle to use when computing free volume
        num_samples (int): Number of samples to use in MC integration

    :py:class`FreeVolume` computes the free volume of a particle assembly using stochastic integration with a test particle type.
    It works together with an HPMC integrator, which defines the particle types used in the simulation.
    As parameters it requires the number of MC integration samples (*nsample*), and the type of particle (*test_type*)
    to use for the integration.


    Examples::

        mc = hoomd.hpmc.integrate.Sphere()
        mc.shape["A"] = {'diameter': 1.0}
        mc.shape["B"] = {'diameter': 0.2}
        mc.depletant_fugacity["B"] = 1.5
        fv = hoomd.hpmc.compute.FreeVolume(test_particle_type='B', num_samples=1000)


    Attributes:
        test_particle_type (str): Type of particle to use when
            computing free volume

        num_samples (int): Number of samples to use in MC
            integration

    """
    def __init__(self, test_particle_type, num_samples):
        # store metadata
        param_dict = ParameterDict(
            test_particle_type=str,
            num_samples=int
        )
        param_dict.update(
            dict(test_particle_type=test_particle_type,
                 num_samples=num_samples))
        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be an HPMC integrator.")

        # Extract 'Shape' from '<hoomd.hpmc.integrate.Shape object>'
        integrator_name = str(integrator).split()[0].split('.')[-1]
        try:
            if isinstance(self._simulation.device, hoomd.device.CPU):
                cpp_cls = getattr(_hpmc, 'ComputeFreeVolume' + integrator_name)
            else:
                cpp_cls = getattr(_hpmc, 'ComputeFreeVolume' + integrator_name + 'GPU')
        except AttributeError:
            raise RuntimeError("Unsupported integrator.\n")

        cl = _hoomd.CellList(self._simulation.state._cpp_sys_def)
        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                integrator._cpp_obj,
                                cl,
                                "")

        super()._attach()

    @log
    def free_volume(self):
        """free volume available to a particle assembly
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.free_volume
        else:
            return None
