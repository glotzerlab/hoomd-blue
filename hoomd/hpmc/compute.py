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
        mc (:py:mod:`hoomd.hpmc.integrate.HPMCIntegrator`): MC integrator.
        type (str): Type of particle to use for integration
        nsample (int): Number of samples to use in MC integration

    :py:class`FreeVolume` computes the free volume of a particle assembly using stochastic integration with a test particle type.
    It works together with an HPMC integrator, which defines the particle types used in the simulation.
    As parameters it requires the number of MC integration samples (*nsample*), and the type of particle (*test_type*)
    to use for the integration.

    A :py:class`FreeVolume` object can be added to a logger for logging during a simulation,
    see :py:class:`hoomd.logging.Logger` for more details.

    Examples::

        mc = hoomd.hpmc.integrate.Sphere()
        mc.shape["A"] = {'diameter': 1.0}
        mc.shape["B"] = {'diameter': 0.2}
        mc.depletant_fugacity["B"] = 1.5
        fv = hoomd.hpmc.compute.FreeVolume(mc=mc, test_type='B', nsample=1000)

    """
    def __init__(self, mc, test_type=None, nsample=None):
        # store metadata
        param_dict = ParameterDict(
            mc=integrate.HPMCIntegrator,
            test_particle_type=str,
            num_samples=int
        )
        param_dict.update(
            dict(mc=mc,
                 test_particle_type=test_type,
                 num_samples=nsample))
        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be an HPMC integrator.")

        integrator_pairs = None
        if isinstance(self._simulation.device, hoomd.device.CPU):
            integrator_pairs = [(integrate.Sphere, _hpmc.ComputeFreeVolumeSphere),
                                (integrate.ConvexPolygon, _hpmc.ComputeFreeVolumeConvexPolygon),
                                (integrate.SimplePolygon, _hpmc.ComputeFreeVolumeSimplePolygon),
                                (integrate.ConvexPolyhedron, _hpmc.ComputeFreeVolumeConvexPolyhedron),
                                (integrate.ConvexSpheropolyhedron, _hpmc.ComputeFreeVolumeSpheropolyhedron),
                                (integrate.Polyhedron, _hpmc.ComputeFreeVolumePolyhedron),
                                (integrate.Ellipsoid, _hpmc.ComputeFreeVolumeEllipsoid),
                                (integrate.ConvexSpheropolygon, _hpmc.ComputeFreeVolumeSpheropolygon),
                                (integrate.FacetedEllipsoid, _hpmc.ComputeFreeVolumeFacetedEllipsoid),
                                (integrate.Sphinx, _hpmc.ComputeFreeVolumeSphinx),
                                (integrate.SphereUnion, _hpmc.ComputeFreeVolumeSphereUnion),
                                (integrate.ConvexSpheropolyhedronUnion, _hpmc.ComputeFreeVolumeConvexPolyhedronUnion),
                                (integrate.FacetedEllipsoidUnion, _hpmc.ComputeFreeVolumeFacetedEllipsoidUnion)]
        else:
            integrator_pairs = [(integrate.Sphere, _hpmc.ComputeFreeVolumeSphereGPU),
                                (integrate.ConvexPolygon, _hpmc.ComputeFreeVolumeConvexPolygonGPU),
                                (integrate.SimplePolygon, _hpmc.ComputeFreeVolumeSimplePolygonGPU),
                                (integrate.ConvexPolyhedron, _hpmc.ComputeFreeVolumeConvexPolyhedronGPU),
                                (integrate.ConvexSpheropolyhedron, _hpmc.ComputeFreeVolumeSpheropolyhedronGPU),
                                (integrate.Polyhedron, _hpmc.ComputeFreeVolumePolyhedronGPU),
                                (integrate.Ellipsoid, _hpmc.ComputeFreeVolumeEllipsoidGPU),
                                (integrate.ConvexSpheropolygon, _hpmc.ComputeFreeVolumeSpheropolygonGPU),
                                (integrate.FacetedEllipsoid, _hpmc.ComputeFreeVolumeFacetedEllipsoidGPU),
                                (integrate.SphereUnion, _hpmc.ComputeFreeVolumeSphereUnionGPU),
                                (integrate.ConvexSpheropolyhedronUnion, _hpmc.ComputeFreeVolumeConvexPolyhedronUnionGPU),
                                (integrate.FacetedEllipsoidUnion, _hpmc.ComputeFreeVolumeFacetedEllipsoidUnionGPU)]
        if isinstance(integrator, integrate.Sphinx) and isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_cls = _hpmc.ComputeFreeVolumeSphinxGPU
        else:
            cpp_cls = None
            for python_integrator, cpp_compute in integrator_pairs:
                if isinstance(integrator, python_integrator):
                    cpp_cls = cpp_compute
            if cpp_cls is None:
                raise RuntimeError("Unsupported integrator.\n")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                self.mc._cpp_obj,
                                _hoomd.CellList(self._simulation.state._cpp_sys_def),
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
