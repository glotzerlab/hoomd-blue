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

# std::string type = m_sysdef->getParticleData()->getNameByType(m_type);
# unsigned int type_int = m_sysdef->getParticleData()->getTypeByName(type);
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

        particle_data = self._simulation.state._cpp_sys_def.getParticleData()
        ntypes = particle_data.getNTypes()
        shape_params = []
        if isinstance(self.shape_params, dict):
            for i in range(particle_data.getNTypes()):
                shape_params.append(self.shape_params[particle_data.getNameByType(i)])
        if isinstance(self.shape_params, list):
            shape_params = self.shape_params
        self._cpp_obj = move_cls(self._simulation.state._cpp_sys_def, ntypes, shape_params)
        self._log_boltzmann_function = boltzmann_cls()
        super()._attach()

    @property
    def shape_params(self):
        if not self._attached:
            return self._param_dict["shape_params"]
        else:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            shape_params = {}
            for i in range(particle_data.getNTypes()):
                shape_params[particle_data.getNameByType(i)] = self._cpp_obj.shape_params[i]
            return shape_params

    @shape_params.setter
    def shape_params(self, new_shape_params):
        # if not self._attached:
        self._param_dict["shape_params"] = new_shape_params
        # else:
        if self._attached:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            for i in range(particle_data.getNTypes()):
                self._cpp_obj.shape_params[i] = new_shape_params[particle_data.getNameByType(i)]


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
                                   stepsize=float(stepsize),
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

        particle_data = self._simulation.state._cpp_sys_def.getParticleData()
        ntypes = particle_data.getNTypes()
        stepsize = []
        if isinstance(self.stepsize, dict):
            for i in range(ntypes):
                stepsize.append(self.stepsize[particle_data.getNameByType(i)])
        elif isinstance(self.stepsize, list):
            stepsize = self.stepsize
        reference = []
        if isinstance(self.reference, dict):
            for i in range(ntypes):
                reference.append(self.reference[particle_data.getNameByType(i)])
        elif isinstance(self.reference, list):
            reference = self.reference
        self._cpp_obj = move_cls(self._simulation.state._cpp_sys_def,
                                 ntypes,
                                 self.stepsize,
                                 self.param_ratio)
        self._log_boltzmann_function = boltzmann_cls(self.stiffness, reference, self._cpp_obj)
        super()._attach()

    @property
    def reference(self):
        if not self._attached:
            return self._param_dict["reference"]
        else:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            reference = {}
            for i in range(particle_data.getNTypes()):
                reference[particle_data.getNameByType(i)] = self._cpp_obj.reference[i]
            return reference

    @reference.setter
    def reference(self, new_reference):
        # if not self._attached:
        self._param_dict["reference"] = new_reference
        # else:
        if self._attached:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            for i in range(particle_data.getNTypes()):
                self._cpp_obj.reference[i] = new_reference[particle_data.getNameByType(i)]

    @property
    def stepsize(self):
        if not self._attached:
            return self._param_dict["stepsize"]
        else:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            stepsize = {}
            for i in range(particle_data.getNTypes()):
                stepsize[particle_data.getNameByType(i)] = self._cpp_obj.stepsize[i]
            return stepsize

    @stepsize.setter
    def stepsize(self, new_stepsize):
        # if not self._attached:
        self._param_dict["stepsize"] = new_stepsize
        # else:
        if self._attached:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            for i in range(particle_data.getNTypes()):
                self._cpp_obj.stepsize[i] = new_stepsize[particle_data.getNameByType(i)]

    @property
    def stiffness(self):
        if self._attached:
            return self._log_boltzmann_function.stiffness(self._simulation.timestep)
        else:
            return self._param_dict["stiffness"]

    @stiffness.setter
    def stiffness(self, new_stiffness):
        self._param_dict["stiffness"] = new_stiffness
        if self._attached:
            self._log_boltzmann_function.stiffness = new_stiffness

    @log(category="scalar")
    def shape_move_stiffness(self):
        """float: Stiffness of the shape used to calculate shape energy

        Returns:
            The stiffness of the shape at the current timestep
        """
        return self.stiffness


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
                                   params=dict(params),
                                   stepsize=dict(stepsize),
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

        particle_data = self._simulation.state._cpp_sys_def.getParticleData()
        ntypes = particle_data.getNTypes()
        stepsize = []
        if isinstance(self.stepsize, dict):
            for i in range(ntypes):
                stepsize.append(self.stepsize[particle_data.getNameByType(i)])
        elif isinstance(self.stepsize, list):
            stepsize = self.stepsize
        params = []
        if isinstance(self.params, dict):
            for i in range(ntypes):
                params.append(self.params[particle_data.getNameByType(i)])
        elif isinstance(self.params, list):
            params = self.params
        self._cpp_obj = move_cls(self._simulation.state._cpp_sys_def,
                                 ntypes,
                                 self.callback,
                                 params,
                                 stepsize,
                                 self.param_ratio)
        self._log_boltzmann_function = boltzmann_cls()
        super()._attach()

    @property
    def stepsize(self):
        if not self._attached:
            return self._param_dict["stepsize"]
        else:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            stepsize = {}
            for i in range(particle_data.getNTypes()):
                stepsize[particle_data.getNameByType(i)] = self._cpp_obj.stepsize[i]
            return stepsize

    @stepsize.setter
    def stepsize(self, new_stepsize):
        # if not self._attached:
        self._param_dict["stepsize"] = new_stepsize
        # else:
        if self._attached:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            for i in range(particle_data.getNTypes()):
                self._cpp_obj.stepsize[i] = new_stepsize[particle_data.getNameByType(i)]

    @property
    def params(self):
        if not self._attached:
            return self._param_dict["params"]
        else:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            params = {}
            for i in range(particle_data.getNTypes()):
                params[particle_data.getNameByType(i)] = self._cpp_obj.params[i]
            return params

    @params.setter
    def params(self, new_params):
        # if not self._attached:
        self._param_dict["params"] = new_params
        # else:
        if self._attached:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            for i in range(particle_data.getNTypes()):
                self._cpp_obj.params[i] = new_params[particle_data.getNameByType(i)]

    @log(category='object')
    def shape_param(self):
        """float: Returns the shape parameter value being used in :py:mod:`python_shape_move`. Returns 0 if another shape move is being used.

        Returns:
            The current value of the shape parameter in the user-specified callback
        """
        return self.params


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
        param_dict = ParameterDict(stepsize=dict(stepsize),
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

        particle_data = self._simulation.state._cpp_sys_def.getParticleData()
        ntypes = particle_data.getNTypes()
        stepsize = []
        if isinstance(self.stepsize, dict):
            for i in range(ntypes):
                stepsize.append(self.stepsize[particle_data.getNameByType(i)])
        elif isinstance(self.stepsize, list):
            stepsize = self.stepsize
        self._cpp_obj = move_cls(self._simulation.state._cpp_sys_def,
                                 ntypes,
                                 stepsize,
                                 self.param_ratio,
                                 self.volume)
        self._log_boltzmann_function = boltzmann_cls()
        super()._attach()

    @property
    def stepsize(self):
        if not self._attached:
            return self._param_dict["stepsize"]
        else:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            stepsize = {}
            for i in range(particle_data.getNTypes()):
                stepsize[particle_data.getNameByType(i)] = self._cpp_obj.stepsize[i]
            return stepsize

    @stepsize.setter
    def stepsize(self, new_stepsize):
        # if not self._attached:
        self._param_dict["stepsize"] = new_stepsize
        # else:
        if self._attached:
            particle_data = self._simulation.state._cpp_sys_def.getParticleData()
            for i in range(particle_data.getNTypes()):
                self._cpp_obj.stepsize[i] = new_stepsize[particle_data.getNameByType(i)]
