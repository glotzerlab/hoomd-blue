# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.operation import _HOOMDBaseObject
from . import _hpmc
from hoomd.hpmc import integrate
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import log
import numpy


class ShapeMove(_HOOMDBaseObject):
    """Base class for all shape moves.

    A shape move is used as an argument to hoomd.hpmc.update.Shape to specify
    how to alter shape definitions

    Note:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    def _attach(self):
        self._apply_param_dict()
        self._apply_typeparam_dict(self._cpp_obj, self._simulation)


class Callback(_HOOMDBaseObject):
    """Base class for callbacks used in Python shape moves.

    Note:
        This class should not be instantiated by users. User-defined callbacks
        should inherit from this class, defining a __call__ method that takes
        a list of floats as an input and returns a shape definition

    Examples::

        class ExampleCallback(hoomd.hpmc.shape_move.Callback):
            def __call__(self, params):
                # do something with params and define verts
                return hoomd.hpmc._hpmc.PolyhedronVertices(verts)
    """

    def __init__(self):
        pass


class Constant(ShapeMove):
    """Apply a transition to a specified shape, changing a particle shape by
    the same way every time the updater is called.

    Note:
        This is useful for calculating a specific transition probability and
        derived thermodynamic quantities.

    Args:
        shape_params (dict): Arguments defining the shape to transition to

    Examples::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        tetrahedron_verts = [(1, 1, 1), (-1, -1, 1),
                             (1, -1, -1), (-1, 1, -1)]
        mc.shape["A"] = dict(vertices=tetrahedron_verts)
        cube_verts = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                      (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)])
        constant_move = hoomd.hpmc.shape_move.Constant(shape_params=cube_verts)

    Attributes:

        shape_params (dict): Arguments defining the shape to transition to

    See Also:
        hoomd.hpmc.integrate for required shape parameters.
    """

    def __init__(self, shape_params):
        self._param_dict.update(ParameterDict(shape_params=dict(shape_params)))

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        move_cls = None
        shapes = [
            'Sphere', 'ConvexPolygon', 'SimplePolygon', 'ConvexPolyhedron',
            'ConvexSpheropolyhedron', 'Ellipsoid', 'ConvexSpheropolygon',
            'Polyhedron', 'Sphinx', 'SphereUnion'
        ]
        for shape in shapes:
            if isinstance(integrator, getattr(integrate, shape)):
                move_cls = getattr(_hpmc, 'ConstantShapeMove' + shape)
        if move_cls is None:
            raise RuntimeError("Integrator not supported")

        ntypes = self._simulation.state._cpp_sys_def.getParticleData(
        ).getNTypes()
        self._cpp_obj = move_cls(self._simulation.state._cpp_sys_def, ntypes,
                                 self.shape_params)
        super()._attach()


class Elastic(ShapeMove):
    """Apply scale and shear shape moves to particles with an energy penalty.

    Args:
        stiffness (Variant): Spring stiffness when shearing particles.

        reference (dict): Arguments defining the shape to reference
            the spring to.

        shear_scale_ratio (float): Fraction of scale to shear moves.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        verts = [(1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1)]
        mc.shape["A"] = dict(vertices=verts)
        elastic_move = hoomd.hpmc.shape_move.Elastic(stiffness=100,
                                                     reference=dict(vertices=verts),
                                                     shear_scale_ratio=0.2)

    Attributes:

        stiffness (Variant): Spring stiffness when shearing particles.

        reference (dict): Arguments defining the shape to reference
            the spring to.

        shear_scale_ratio (float): Fraction of scale to shear moves.
    """

    def __init__(self, stiffness, reference, shear_scale_ratio):
        param_dict = ParameterDict(stiffness=hoomd.variant.Variant,
                                   reference=dict(reference),
                                   shear_scale_ratio=float(shear_scale_ratio))
        param_dict["stiffness"] = stiffness
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        move_cls = None
        if isinstance(integrator, integrate.ConvexPolyhedron):
            move_cls = _hpmc.ElasticShapeMoveConvexPolyhedron
        elif isinstance(integrator, integrate.Ellipsoid):
            move_cls = _hpmc.ElasticShapeMoveEllipsoid
            for type_shape in self.mc.type_shapes():
                if not numpy.isclose(type_shape["a"], type_shape["b"]) or \
                   not numpy.isclose(type_shape["a"], type_shape["c"]) or \
                   not numpy.isclose(type_shape["b"], type_shape["c"]):
                    raise ValueError(
                        "This updater only works when a=b=c initially.")
        else:
            raise RuntimeError("Integrator not supported")

        ntypes = self._simulation.state._cpp_sys_def.getParticleData(
        ).getNTypes()
        self._cpp_obj = move_cls(self._simulation.state._cpp_sys_def, ntypes,
                                 self.shear_scale_ratio, self.stiffness,
                                 self.reference)
        super()._attach()


class Python(ShapeMove):
    """Apply custom shape moves to particles through a Python callback.

    Args:
        callback (Callback): The python class that will be called
            to update the particle shapes

        params (dict): Dictionary of types and the corresponding list
            of initial parameters to pass to the callback
            (ex: {'A' : [1.0], 'B': [0.0]})

        param_move_probability (float): Average fraction of parameters to change during
            each shape move

    Note:
        Parameters must be given for every particle type. The callback should
        rescale the particle to have constant volume if desired.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        mc.shape["A"] = dict(vertices=[(1, 1, 1), (-1, -1, 1),
                                       (1, -1, -1), (-1, 1, -1)])
        # example callback
        class ExampleCallback(hoomd.hpmc.shape_move.Callback):
            def __call__(self, params):
                # do something with params and define verts
                return hoomd.hpmc._hpmc.PolyhedronVertices(verts)
        python_move = hoomd.hpmc.shape_move.Python(callback=ExampleCallback,
                                                   params={'A': [1.0]},
                                                   param_move_probability=1.0)

    Attributes:

        callback (Callback): The python class that will be called
            to update the particle shapes

        params (dict): Dictionary of types and the corresponding list
            of initial parameters to pass to the callback
            (ex: {'A' : [1.0], 'B': [0.0]})

        param_move_probability (float): Average fraction of parameters to change during
            each shape move
    """

    def __init__(self, callback, params, stepsize, param_move_probability):
        param_dict = ParameterDict(
            callback=Callback,
            params=dict(params),
            param_move_probability=float(param_move_probability))
        param_dict["callback"] = callback
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        move_cls = None
        shapes = [
            'Sphere', 'ConvexPolygon', 'SimplePolygon', 'ConvexPolyhedron',
            'ConvexSpheropolyhedron', 'Ellipsoid', 'ConvexSpheropolygon',
            'Polyhedron', 'Sphinx', 'SphereUnion'
        ]
        print()
        for shape in shapes:
            if isinstance(integrator, getattr(integrate, shape)):
                move_cls = getattr(_hpmc, 'PythonShapeMove' + shape)
        if move_cls is None:
            raise RuntimeError("Integrator not supported")

        ntypes = self._simulation.state._cpp_sys_def.getParticleData(
        ).getNTypes()
        self._cpp_obj = move_cls(self._simulation.state._cpp_sys_def, ntypes,
                                 self.callback, self.params,
                                 self.param_move_probability)
        super()._attach()

    @log(category='object')
    def shape_param(self):
        """float: Shape parameter values being used.

        None when not attached
        """
        return self.params


class Vertex(ShapeMove):
    """Apply shape moves where particle vertices are translated.

    Args:

        vertex_move_probability (float): Average fraction of vertices to change during
            each shape move

        volume (float): Volume of the particles to hold constant

    Note:
        Vertices are rescaled during each shape move to ensure that the shape
        maintains a constant volume

    Note:
        The shape definition used corresponds to the convex hull of the
        vertices.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        cube_verts = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                      (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)])
        mc.shape["A"] = dict(vertices=numpy.asarray(cube_verts) / 2)
        vertex_move = hoomd.hpmc.shape_move.Vertex(vertex_move_probability=0.125,
                                                   volume=1.0)

    Attributes:

        vertex_move_probability (float): Average fraction of vertices to change during
            each shape move

        volume (float): Volume of the particles to hold constant
    """

    def __init__(self, vertex_move_probability, volume):
        param_dict = ParameterDict(
            vertex_move_probability=float(vertex_move_probability),
            volume=float(volume))
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        move_cls = None
        if isinstance(integrator, integrate.ConvexPolyhedron):
            move_cls = _hpmc.GeneralizedShapeMoveConvexPolyhedron
        else:
            raise RuntimeError("Integrator not supported")

        ntypes = self._simulation.state._cpp_sys_def.getParticleData(
        ).getNTypes()
        self._cpp_obj = move_cls(self._simulation.state._cpp_sys_def, ntypes,
                                 self.vertex_move_probability, self.volume)
        super()._attach()
