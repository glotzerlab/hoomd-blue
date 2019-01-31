# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Compute properties of hard particle configurations.
"""

from __future__ import print_function

from hoomd import _hoomd
from hoomd.hpmc import _hpmc
from hoomd.hpmc import integrate
from hoomd.compute import _compute
import hoomd

class free_volume(_compute):
    R""" Compute the free volume available to a test particle by stochastic integration.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        seed (int): Random seed for MC integration.
        type (str): Type of particle to use for integration
        nsample (int): Number of samples to use in MC integration
        suffix (str): Suffix to use for log quantity

    :py:class`free_volume` computes the free volume of a particle assembly using stochastic integration with a test particle type.
    It works together with an HPMC integrator, which defines the particle types used in the simulation.
    As parameters it requires the number of MC integration samples (*nsample*), and the type of particle (*test_type*)
    to use for the integration.

    Once initialized, the compute provides a log quantity
    called **hpmc_free_volume**, that can be logged via :py:class:`hoomd.analyze.log`.
    If a suffix is specified, the log quantities name will be
    **hpmc_free_volume_suffix**.

    Examples::

        mc = hpmc.integrate.sphere(seed=415236)
        compute.free_volume(mc=mc, seed=123, test_type='B', nsample=1000)
        log = analyze.log(quantities=['hpmc_free_volume'], period=100, filename='log.dat', overwrite=True)

    """
    def __init__(self, mc, seed, suffix='', test_type=None, nsample=None):
        hoomd.util.print_status_line();

        # initialize base class
        _compute.__init__(self);

        # create the c++ mirror class
        cl = _hoomd.CellList(hoomd.context.current.system_definition);
        hoomd.context.current.system.addCompute(cl, "auto_cl3")

        cls = None;
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.ComputeFreeVolumeSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.ComputeFreeVolumeConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.ComputeFreeVolumeSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _hpmc.ComputeFreeVolumeConvexPolyhedron;
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.ComputeFreeVolumeSpheropolyhedron;
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.ComputeFreeVolumeEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.ComputeFreeVolumeSpheropolygon;
            elif isinstance(mc, integrate.faceted_ellipsoid):
                cls =_hpmc.ComputeFreeVolumeFacetedEllipsoid;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.ComputeFreeVolumePolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.ComputeFreeVolumeSphinx;
            elif isinstance(mc, integrate.convex_polyhedron_union):
                cls = _hpmc.ComputeFreeVolumeConvexPolyhedronUnion
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls = _hpmc.ComputeFreeVolumeFacetedEllipsoidUnion
            elif isinstance(mc, integrate.sphere_union):
                cls = _hpmc.ComputeFreeVolumeSphereUnion;
            else:
                hoomd.context.msg.error("compute.free_volume: Unsupported integrator.\n");
                raise RuntimeError("Error initializing compute.free_volume");
        else:
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.ComputeFreeVolumeGPUSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.ComputeFreeVolumeGPUConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.ComputeFreeVolumeGPUSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _hpmc.ComputeFreeVolumeGPUConvexPolyhedron;
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.ComputeFreeVolumeGPUSpheropolyhedron;
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.ComputeFreeVolumeGPUEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.ComputeFreeVolumeGPUSpheropolygon;
            elif isinstance(mc, integrate.faceted_ellipsoid):
                cls =_hpmc.ComputeFreeVolumeGPUFacetedEllipsoid;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.ComputeFreeVolumeGPUPolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.ComputeFreeVolumeGPUSphinx;
            elif isinstance(mc, integrate.sphere_union):
                cls = _hpmc.ComputeFreeVolumeGPUSphereUnion;
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls = _hpmc.ComputeFreeVolumeGPUFacetedEllipsoidUnion;
            elif isinstance(mc, integrate.convex_polyhedron_union):
                cls = _hpmc.ComputeFreeVolumeGPUConvexPolyhedronUnion;
            else:
                hoomd.context.msg.error("compute.free_volume: Unsupported integrator.\n");
                raise RuntimeError("Error initializing compute.free_volume");

        if suffix is not '':
            suffix = '_' + suffix

        self.cpp_compute = cls(hoomd.context.current.system_definition,
                                mc.cpp_integrator,
                                cl,
                                seed,
                                suffix)

        if test_type is not None:
            itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(test_type)
            self.cpp_compute.setTestParticleType(itype)
        if nsample is not None:
            self.cpp_compute.setNumSamples(int(nsample))

        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name)
        self.enabled = True
