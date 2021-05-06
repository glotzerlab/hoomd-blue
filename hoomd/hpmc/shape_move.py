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