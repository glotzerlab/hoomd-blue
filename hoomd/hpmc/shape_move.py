# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Shape moves for a for alchemical simulations in extended ensembles."""

import hoomd
from hoomd.operation import _HOOMDBaseObject
from . import _hpmc
from hoomd.hpmc import integrate
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
import numpy


class ShapeMove(_HOOMDBaseObject):
    """Base class for all shape moves.

    Args:
        move_probability (`float`): Probability of performing a shape move.
        See the  documentation of each derived class for a
            descrition of how this parameter is interpreted in the context of
            each shape move subclass.

    Note:
        See the documentation of each derived class for a list of supported
        shapes.

    Note:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    _suported_shapes = None

    def __init__(self, move_probability):
        # Set base parameter dict for all shape_moves
        param_dict = ParameterDict(move_probability=float(move_probability))
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        integrator_name = integrator.__class__.__name__
        if integrator_name in self._suported_shapes:
            self._move_cls = getattr(_hpmc,
                                     self.__class__.__name__ + integrator_name)
        else:
            raise RuntimeError("Integrator not supported")
        self._cpp_obj = self._move_cls(
            self._simulation.state._cpp_sys_def,
            self._simulation.operations.integrator._cpp_obj)
        super()._attach()


class Elastic(ShapeMove):
    """Apply scale and shear shape moves to particles with an energy penalty.

    Args:
        stiffness (:py:mod:`hoomd.variant.Variant`): Shape stiffness against
            deformations.

        mc (`type` or `hoomd.hpmc.integrate.HPMCIntegrator`): The class of
            the MC shape integrator or an instance (see `hoomd.hpmc.integrate`)
            to use with this elastic shape. Must be a compatible class.

        move_probability (`float`, optional): Fraction of scale to shear
            moves (**default**: 0.5).

    .. rubric:: Shape support.

    The following shapes are supported:
        * `hoomd.hpmc.integrate.ConvexPolyhedron`
        * `hoomd.hpmc.integrate.Ellipsoid`

    Note:
        An instance is only able to be used with the passed HPMC integrator
        class.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron()
        verts = [(1, 1, 1), (-1, -1, 1), (1, -1, -1), (-1, 1, -1)]
        mc.shape["A"] = dict(vertices=verts)
        elastic_move = hoomd.hpmc.shape_move.Elastic(stiffness=10, mc)
        elastic_move.stiffness = 100
        elastic_move.reference_shape["A"] = verts

    Attributes:
        stiffness (:py:mod:`hoomd.variant.Variant`): Shape stiffness against
            deformations.

        reference_shape (`TypeParameter` [``particle type``, `dict`]): Reference
            shape against to which compute the deformation energy.

        move_probability (`float`): Fraction of scale to shear moves.
    """

    _suported_shapes = {'ConvexPolyhedron', 'Ellipsoid'}

    def __init__(self, stiffness, mc, move_probability=0.5):

        super().__init__(move_probability)
        param_dict = ParameterDict(stiffness=hoomd.variant.Variant)
        param_dict["stiffness"] = stiffness
        self._param_dict.update(param_dict)
        self._add_typeparam(self._get_shape_param(mc))

    def _get_shape_param(self, mc):
        if isinstance(mc, hoomd.hpmc.integrate.HPMCIntegrator):
            cls = mc.__class__
        else:
            cls = mc
        if cls.__name__ not in self._suported_shapes:
            raise ValueError(f"Unsupported integrator type {cls}. Supported "
                             f"types are {self._suported_shapes}")
        # Class can only be used for this type of integrator now.
        self._suported_shapes = {cls.__name__}
        shape = cls().shape
        shape.name = "reference_shape"
        return shape

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if isinstance(integrator, integrate.Ellipsoid):
            for shape in integrator.shape.values():
                if not numpy.allclose((shape["a"], shape["b"], shape["c"]), shape["a"]):
                    raise ValueError("This updater only works when a=b=c.")
        super()._attach()


class ShapeSpace(ShapeMove):
    """Apply shape moves in a :math:`N` dimensional shape space.

    Args:
        callback (``callable`` [`str`, `list`], `dict` ]): The python function
            that will be called to map the given shape parameters to a shape
            definition. The function takes the particle type and a list of
            parameters as arguments and return a dictionary with the shape
            definition whose keys **must** match the shape definition of the
            integrator:

                ``callable[[str, list], dict]``

            Note that there is no type validation of the callback.

        move_probability (`float`, optional): Average fraction of shape
            parameters to change each timestep (**default**: 1).

    .. rubric:: Shape support.

    The following shapes are supported:
        * `hoomd.hpmc.integrate.ConvexPolyhedron`
        * `hoomd.hpmc.integrate.ConvexSpheropolyhedron`
        * `hoomd.hpmc.integrate.Ellipsoid`

    Attention:
        The acceptance criteria of shape moves requires computing the particle's
        moment of inertia before and after the trial moves. Currently, computing
        the moments of inertia for spheropolyhedra is not fully implemented.
        However, the use of this shape move with spheropolyhedra is currently
        enabled to allow the use of spherical depletants with shape moves.

    Note:
        Parameters must be given for every particle type and must be between 0
        and 1. This class is limited to performing MC moves on the predefined
        shape parameters, it does not performs any consistency checks
        internally. Therefore, any shape constraint (e.g. constant volume, etc)
        must be performed within the callback.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron()
        mc.shape["A"] = dict(vertices=[(1, 1, 1), (-1, -1, 1),
                                       (1, -1, -1), (-1, 1, -1)])
        # example callback
        class ExampleCallback:
            def __init__(self):
                default_dict = dict(sweep_radius=0, ignore_statistics=True)
            def __call__(self, type, param_list):
                # do something with params and define verts
                return dict("vertices":verts, **self.default_dict))
        move = hpmc.shape_move.ShapeSpace(callback = ExampleCallback)

    Attributes:
        callback (``callable`` [`str`, `list`], `dict` ]): The python function
            that will be called to map the given shape parameters to a shape
            definition. The function takes the particle type and a list of
            parameters as arguments and return a dictionary with the shape
            definition whose keys **must** match the shape definition of the
            integrator:

                ``callable[[str, list], dict]``

        params (`TypeParameter` [``particle type``, `list`]): List of tunable
            parameters to be updated. The length of the list defines the
            dimension of the shape space for each particle type.

        move_probability (`float`, optional): Average fraction of shape
            parameters to change each timestep (**default**: 1).
    """

    _suported_shapes = {
        'ConvexPolyhedron', 'ConvexSpheropolyhedron', 'Ellipsoid'
    }

    def __init__(self, callback, move_probability=1):

        super().__init__(move_probability)
        param_dict = ParameterDict(callback=object)
        param_dict["callback"] = callback
        self._param_dict.update(param_dict)

        typeparam_shapeparams = TypeParameter('params',
                                              type_kind='particle_types',
                                              param_dict=TypeParameterDict(
                                                  [float], len_keys=1))
        self._add_typeparam(typeparam_shapeparams)


class Vertex(ShapeMove):
    """Apply shape moves where particle vertices are translated.

    Args:
        move_probability (`float`, optional): Average fraction of
            vertices to change during each shape move (**default**: 1).

    .. rubric:: Shape support.

    The following shapes are supported:
        * `hoomd.hpmc.integrate.ConvexPolyhedron`

    Note:
        Vertices are rescaled during each shape move to ensure that the shape
        maintains a constant volume. To preserve detail balance, the maximum
        step size is rescaled by volume**(1/3) every time a move is accepted.

    Note:
        The shape definition used corresponds to the convex hull of the
        vertices.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        cube_verts = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                      (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)])
        mc.shape["A"] = dict(vertices=numpy.asarray(cube_verts) / 2)
        vertex_move = shape_move.Vertex()
        vertex_move.volume["A"] = 1

    Attributes:
        move_probability (`float`): Average fraction of vertices to change
            during each shape move.

        volume (`TypeParameter` [``particle type``, `float`]): Volume of the
            particles to hold constant.
    """

    _suported_shapes = {'ConvexPolyhedron'}

    def __init__(self, move_probability=1):

        super().__init__(move_probability)

        typeparam_volume = TypeParameter('volume',
                                         type_kind='particle_types',
                                         param_dict=TypeParameterDict(
                                             float, len_keys=1))
        self._add_typeparam(typeparam_volume)
