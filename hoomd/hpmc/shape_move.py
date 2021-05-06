import hoomd
from hoomd.operation import _HOOMDBaseObject
from . import _hpmc
from hoomd.hpmc import integrate
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import log


class ShapeMove(_HOOMDBaseObject):
    def _attach(self):
        self._apply_param_dict()
        self._apply_typeparam_dict(self._cpp_obj, self._simulation)


class Constant(ShapeMove):
    R"""
    Enable constant shape move and set parameters. Changes a particle shape by
    the same way every time the updater is called. This is useful for calculating
    a specific transition probability and derived thermodynamic quantities.

    Args:
        shape_params: Arguments required to define the :py:mod:`hoomd.hpmc.integrate' reference shape. 

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        mc.shape["A"] = dict(vertices=[(1, 1, 1), (-1, -1, 1), (1, -1, -1),
                                       (-1, 1, -1)])
        shape_up = hpmc.update.Alchemy(mc, move_ratio=0.25, seed=9876)
        # convex_polyhedron
        shape_up.constant_shape_move(vertices=[(1, 1, 1), (-1, -1, 1), (1, -1, -1),
                                               (-1, 1, -1)])

    See Also:
        :py:mod:`hoomd.hpmc.integrate` for required shape parameters.

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
        boltzmann_cls = None
        shapes = ['Sphere', 'ConvexPolygon', 'SimplePolygon',
                  'ConvexPolyhedron', 'ConvexSpheropolyhedron',
                  'Ellipsoid', 'ConvexSpheropolygon', 'Polyhedron',
                  'Sphinx', 'SphereUnion']
        for shape in shapes:
            if isinstance(integrator, getattr(integrate, shape)):
                move_cls = getattr(_hpmc, 'ConstantShapeMove' + shape)
                boltzmann_cls = getattr(_hpmc, 'AlchemyLogBoltzmann' + shape)
        if move_cls is None or boltzmann_cls is None:
            raise RuntimeError("Integrator not supported")

        ntypes = len(integrator.state["shape"].keys()) - 1
        self._cpp_obj = move_cls(ntypes, self.shape_params)
        self._boltzmann_function = boltzmann_cls()
        super()._attach()


class Elastic(ShapeMove):
    R"""
    Enable scale and shear shape move and set parameters. Changes a particle shape by
    scaling the particle and shearing the particle.

    Args:
        stepsize (float): Largest scaling/shearing factor used.
        param_ratio (float): Fraction of scale to shear moves.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        mc.shape["A"] = dict(vertices=[(1, 1, 1), (-1, -1, 1), (1, -1, -1),
                                       (-1, 1, -1)])
        shape_up = hpmc.update.Alchemy(mc, param_ratio=0.25, seed=9876)
        shape_up.elastic_shape_move(stepsize=0.01)

    """
    def __init__(self, stiffness, reference, stepsize, param_ratio):
        param_dict = ParameterDict(stiffness=hoomd.variant.Variant,
                                   reference=dict(reference),
                                   stepsize=list(stepsize),
                                   param_ratio=float(param_ratio))
        param_dict["stiffness"] = stiffness
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        move_cls = None
        shape_cls = None
        boltzmann_cls = None
        if isinstance(integrator, integrate.ConvexPolyhedron):
            move_cls = _hpmc.ElasticShapeMoveConvexPolyhedron
            boltzmann_cls = _hpmc.ShapeSpringLogBoltzmannConvexPolyhedron
            shape_cls = hoomd.hpmc._hpmc.PolyhedronVertices
        elif isinstance(integrator, integrate.Ellipsoid):
            move_cls = _hpmc.ElasticShapeMoveEllipsoid
            for type_shape in self.mc.type_shapes():
                if not np.isclose(type_shape["a"], type_shape["b"]) or \
                   not np.isclose(type_shape["a"], type_shape["c"]) or \
                   not np.isclose(type_shape["b"], type_shape["c"]):
                    raise ValueError("This updater only works when a=b=c initially.")
            boltzmann_cls = _hpmc.ShapeSpringLogBoltzmannEllipsoid
            shape_cls = hoomd.hpmc._hpmc.EllipsoidParams
        else:
            raise RuntimeError("Integrator not supported")

        ntypes = len(self.mc.state["shape"].keys()) - 1
        self._shape_move = move_cls(ntypes, self.stepsize, self.param_ratio)
        ref_shape = shape_cls(self.reference)
        self._boltzmann_function = boltzmann_cls(self.stiffness, ref_shape, self._cpp_obj)
        super()._attach()

    @property
    def stiffness(self):
        return self._boltzmann_function.stiffness(self._simulation.timestep)

    @stiffness.setter
    def stiffness(self, new_stiffness):
        self._param_dict["stiffness"] = new_stiffness
        self._boltzmann_function.stiffness = new_stiffness

    @log(category="scalar")
    def shape_move_stiffness(self):
        """float: Stiffness of the shape used to calculate shape energy

        Returns:
            The stiffness of the shape at the current timestep
        """
        return self.stiffness

    @log(category="scalar")
    def shape_move_energy(self):
        """float: Energy of the shape resulting from shear moves

        Returns:
            The energy of the shape at the current timestep
        """
        return sum([self._cpp_obj.getShapeMoveEnergy(i, self._simulation.timestep) for i in range(self._simulation.state._cpp_sys_def.getParticleData().getNTypes())])


class Python(ShapeMove):
    R"""Enable python shape move and) set parameters.
    All python shape moves must be callable object that take a single list
    of parameters between 0 and 1 as the call arguments and returns a
    shape parameter definition.

    Args:
        callback (callable): The python function that will be called each update.
        params (dict): Dictionary of types and the corresponding list parameters (ex: {'A' : [1.0], 'B': [0.0]})
        stepsize (float): Step size in parameter space.
        param_ratio (float): Average fraction of parameters to change each update

    Note:
        Parameters must be given for every particle type. Callback should rescale the particle to have constant
        volume if necessary/desired.

    Example::

        # example callback
        class convex_polyhedron_callback:
            def __init__(self, mc):
                self.mc = mc;
            def __call__(self, params):
                # do something with params and define verts
                return hoomd.hpmc._hpmc.PolyhedronVertices(verts)
        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        mc.shape["A"] = dict(vertices=[(1, 1, 1), (-1, -1, 1), (1, -1, -1),
                                       (-1, 1, -1)])

        # now set up the updater
        shape_up = hpmc.update.Alchemy(mc, move_ratio=0.25, seed=9876)
        shape_up.python_shape_move(callback=convex_polyhedron_callback(mc), params={'A': [0.5]}, stepsize=0.001, param_ratio=0.5)

    """
    def __init__(self, callback, params, stepsize, param_ratio):
        param_dict = ParameterDict(callback=callable,
                                   params=list(params),
                                   stepsize=list(stepsize),
                                   param_ratio=float(param_ratio))
        param_dict["callback"] = callback
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        move_cls = None
        boltzmann_cls = None
        shapes = ['Sphere', 'ConvexPolygon', 'SimplePolygon',
                  'ConvexPolyhedron', 'ConvexSpheropolyhedron',
                  'Ellipsoid', 'ConvexSpheropolygon', 'Polyhedron',
                  'Sphinx', 'SphereUnion']
        for shape in shapes:
            if isinstance(integrator, getattr(integrate, shape)):
                move_cls = getattr(_hpmc, 'PythonShapeMove' + shape)
                boltzmann_cls = getattr(_hpmc, 'AlchemyLogBoltzmann' + shape)
        if move_cls is None or boltzmann_cls is None:
            raise RuntimeError("Integrator not supported")

        ntypes = len(self.mc.state["shape"].keys()) - 1
        self._cpp_obj = move_cls(ntypes, self.callback, self.params,
                                 self.stepsize, self.param_ratio)
        self._boltzmann_function = boltzmann_cls()
        super()._attach()

    @log(category='scalar')
    def shape_param(self):
        """float: Returns the shape parameter value being used in :py:mod:`python_shape_move`. Returns 0 if another shape move is being used.

        Returns:
            The current value of the shape parameter in the user-specified callback
        """
        return self._cpp_obj.getShapeParam("shape_param-0", self._simulation.timestep)


class Vertex(ShapeMove):
    R"""
    Enable vertex shape move and set parameters. Changes a particle shape by
    translating vertices and rescaling to have constant volume. The shape definition
    corresponds to the convex hull of the vertices.

    Args:
        stepsize (float): Stepsize for each vertex move
        param_ratio (float): Average fraction of vertices to change each update
        volume (float): Volume of the particles to hold constant

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
        mc.shape["A"] = dict(vertices=[(1, 1, 1), (-1, -1, 1), (1, -1, -1),
                                       (-1, 1, -1)])
        shape_up = hpmc.update.Alchemy(mc, move_ratio=0.25, seed=9876)
        shape_up.vertex_shape_move(stepsize=0.001, param_ratio=0.25, volume=1.0)

    """
    def __init__(self, stepsize, param_ratio, volume):
        param_dict = ParameterDict(stepsize=list(stepsize),
                                   param_ratio=float(param_ratio),
                                   volume=float(volume))
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        move_cls = None
        boltzmann_cls = None
        if isinstance(integrator, integrate.ConvexPolyhedron):
            move_cls = _hpmc.GeneralizedShapeMoveConvexPolyhedron
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannConvexPolyhedron
        else:
            raise RuntimeError("Integrator not supported")

        ntypes = len(self.mc.state["shape"].keys()) - 1
        self._cpp_obj = move_cls(ntypes, self.stepsize,
                                 self.param_ratio, self.volume)
        self._boltzmann_function = boltzmann_cls()
        super()._attach()
