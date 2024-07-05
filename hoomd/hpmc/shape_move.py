# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Shape moves for a for alchemical simulations in extended ensembles.

`ShapeMove` subclasses extend the Hamiltonian of the system by adding degrees of
freedom to the shape of the hard particle.

See Also:
    * `G. van Anders <https://doi.org/10.1021/acsnano.5b04181>`_
    * `Y. Geng <https://doi.org/10.1126/sciadv.aaw0514>`_
"""

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
        default_step_size (`float`, optional): Default maximum size of shape
            trial moves (**default**: None). By default requires setting step
            size for all types.

    Note:
        See the documentation of each derived class for a list of supported
        shapes.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

    Attributes:
        step_size (`TypeParameter` [``particle type``, `float`]): Maximum size
            of shape trial moves.
    """

    _suported_shapes = None

    def __init__(self, default_step_size=None):
        if default_step_size is None:
            step_size = float
        else:
            step_size = float(default_step_size)
        typeparam_step_size = TypeParameter('step_size',
                                            type_kind='particle_types',
                                            param_dict=TypeParameterDict(
                                                step_size, len_keys=1))
        self._add_typeparam(typeparam_step_size)

    def _attach_hook(self):
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


class Elastic(ShapeMove):
    """Elastic shape move.

    Apply volume preserving normal and shear shape moves to particles with a
    deformation energy penalty. When volume preserving deformations, the strain
    energy penalty is approximated as:

    .. math::

        U \\approx \\mu \\text{Tr}[\\boldsymbol{\\varepsilon}^2]V

    where :math:`\\mu` is the stiffness constant, :math:`\\text{Tr}` is the
    trace operator, :math:`\\boldsymbol{\\varepsilon}` is the strain tensor
    and :math:`V` is the particle volume.

    Args:
        stiffness (hoomd.variant.variant_like): Shape stiffness against
            deformations.

        mc (`type` or `hoomd.hpmc.integrate.HPMCIntegrator`): The class of
            the MC shape integrator or an instance (see `hoomd.hpmc.integrate`)
            to use with this elastic shape. Must be a compatible class. We use
            this argument to create validation for `reference_shape`.

        default_step_size (`float`, optional): Default maximum size of shape
            trial moves (**default**: None). By default requires setting step
            size for all types.

        normal_shear_ratio (`float`, optional): Fraction of normal to shear
            deformation trial moves (**default**: 0.5).

    .. rubric:: Shape support.

    The following shapes are supported:

    * `hoomd.hpmc.integrate.ConvexPolyhedron`

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
        stiffness (hoomd.variant.Variant): Shape stiffness against
            deformations.

        step_size (`TypeParameter` [``particle type``, `float`]): Maximum size
            of shape trial moves.

        reference_shape (`TypeParameter` [``particle type``, `dict`]): Reference
            shape against to which compute the deformation energy.

        normal_shear_ratio (`float`, optional): Fraction of normal to shear
            deformation trial moves (**default**: 0.5).
    """

    _suported_shapes = {'ConvexPolyhedron'}

    def __init__(self,
                 stiffness,
                 mc,
                 default_step_size=None,
                 normal_shear_ratio=0.5):
        super().__init__(default_step_size)
        param_dict = ParameterDict(normal_shear_ratio=float(normal_shear_ratio),
                                   stiffness=hoomd.variant.Variant)
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

    def _attach_hook(self):
        integrator = self._simulation.operations.integrator
        if isinstance(integrator, integrate.Ellipsoid):
            for shape in integrator.shape.values():
                is_sphere = numpy.allclose((shape["a"], shape["b"], shape["c"]),
                                           shape["a"])
                if not is_sphere:
                    raise ValueError("This updater only works when a=b=c.")
        super()._attach_hook()


class ShapeSpace(ShapeMove):
    """Apply shape moves in a :math:`N` dimensional shape space.

    Args:
        callback (``callable`` [`str`, `list`], `dict` ]): The python callable
            that will be called to map the given shape parameters to a shape
            definition. The callable takes the particle type and a list of
            parameters as arguments and return a dictionary with the shape
            definition whose keys **must** match the shape definition of the
            integrator: ``callable[[str, list], dict]``. There is no
            type validation of the callback.

        default_step_size (`float`, optional): Default maximum size of shape
            trial moves (**default**: None). By default requires setting step
            size for all types.

        param_move_probability (`float`, optional): Average fraction of shape
            parameters to change each timestep (**default**: 1).

    .. rubric:: Shape support.

    The following shapes are supported:

    * `hoomd.hpmc.integrate.ConvexPolyhedron`
    * `hoomd.hpmc.integrate.ConvexSpheropolyhedron`
    * `hoomd.hpmc.integrate.Ellipsoid`

    Attention:
        `ShapeSpace` only works with spheropolyhedra that have zero rounding or
        are spheres.

    Note:
        Parameters must be given for every particle type and must be between 0
        and 1. The class does not performs any consistency checks internally.
        Therefore, any shape constraint (e.g. constant volume, etc) must be
        performed within the callback.

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
            integrator: ``callable[[str, list], dict]``.

        step_size (`TypeParameter` [``particle type``, `float`]): Maximum size
            of shape trial moves.

        params (`TypeParameter` [``particle type``, `list`]): List of tunable
            parameters to be updated. The length of the list defines the
            dimension of the shape space for each particle type.

        param_move_probability (`float`, optional): Average fraction of shape
            parameters to change each timestep (**default**: 1).
    """

    _suported_shapes = {
        'ConvexPolyhedron', 'ConvexSpheropolyhedron', 'Ellipsoid'
    }

    def __init__(self,
                 callback,
                 default_step_size=None,
                 param_move_probability=1):
        super().__init__(default_step_size)
        param_dict = ParameterDict(
            param_move_probability=float(param_move_probability),
            callback=object)
        param_dict["callback"] = callback
        self._param_dict.update(param_dict)

        typeparam_shapeparams = TypeParameter('params',
                                              type_kind='particle_types',
                                              param_dict=TypeParameterDict(
                                                  [float], len_keys=1))
        self._add_typeparam(typeparam_shapeparams)


class Vertex(ShapeMove):
    """Apply shape moves where particle vertices are translated.

    Vertices are rescaled during each shape move to ensure that the shape
    maintains a constant volume. To preserve detail balance, the maximum step
    size is rescaled by volume**(1/3) every time a move is accepted.

    Args:
        default_step_size (`float`, optional): Default maximum size of shape
            trial moves (**default**: None). By default requires setting step
            size for all types.

        vertex_move_probability (`float`, optional): Average fraction of
            vertices to change during each shape move (**default**: 1).

    `Vertex` moves apply a uniform move on each vertex with probability
    `vertex_move_probability` in a shape up to a maximum displacement of
    `step_size` for the composing instance. The shape volume is rescaled at the
    end of all the displacements to the specified volume. To preserve detail
    balance, the maximum step size is rescaled by :math:`V^{1 / 3}` every time
    a move is accepted.

    .. rubric:: Shape support.

    The following shapes are supported:

    * `hoomd.hpmc.integrate.ConvexPolyhedron`

    Note:
        `hoomd.hpmc.integrate.ConvexPolyhedron` models shapes that are the the
        convex hull of the given vertices.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        cube_verts = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
                      (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)])
        mc.shape["A"] = dict(vertices=numpy.asarray(cube_verts) / 2)
        vertex_move = shape_move.Vertex()
        vertex_move.volume["A"] = 1

    Attributes:
        step_size (`TypeParameter` [``particle type``, `float`]): Maximum size
            of shape trial moves.

        vertex_move_probability (`float`): Average fraction of vertices to
            change during each shape move.

        volume (`TypeParameter` [``particle type``, `float`]): Volume of the
            particles to hold constant. The particles size is always rescaled to
            maintain this volume.
    """

    _suported_shapes = {'ConvexPolyhedron'}

    def __init__(self, default_step_size=None, vertex_move_probability=1):
        super().__init__(default_step_size)
        param_dict = ParameterDict(
            vertex_move_probability=float(vertex_move_probability))
        self._param_dict.update(param_dict)
        typeparam_volume = TypeParameter('volume',
                                         type_kind='particle_types',
                                         param_dict=TypeParameterDict(
                                             float, len_keys=1))
        self._add_typeparam(typeparam_volume)
