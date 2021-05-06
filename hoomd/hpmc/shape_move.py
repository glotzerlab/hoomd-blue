import hoomd
from hoomd.operation import _HOOMDBaseObject
from . import _hpmc
from hoomd.hpmc import integrate
from hoomd.data.parameterdicts import ParameterDict


class ShapeMove(_HOOMDBaseObject):
    def _attach(self):
        self._apply_param_dict()
        self._apply_typeparam_dict(self._cpp_obj, self._simulation)


class Constant(ShapeMove):
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

        ntypes = len(integrator.state["shape"].keys()) - 1
        self._cpp_obj = move_cls(ntypes, self.shape_params)
        self._boltzmann_function = boltzmann_cls()
        super()._attach()


class Elastic(ShapeMove):
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
        return self._boltzmann_function.stiffness

    @stiffness.setter
    def stiffness(self, new_stiffness):
        self._param_dict["stiffness"] = new_stiffness
        self._boltzmann_function.stiffness = new_stiffness


class Python(ShapeMove):
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

        ntypes = len(self.mc.state["shape"].keys()) - 1
        self._cpp_obj = move_cls(ntypes, self.callback, self.params,
                                 self.stepsize, self.param_ratio)
        self._boltzmann_function = boltzmann_cls()
        super()._attach()