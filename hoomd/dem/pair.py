import hoomd;
import hoomd.md;
import hoomd.md.nlist as nl;

from math import sqrt

from . import _dem
from . import params
from . import utils

class _DEMBase:
    def __init__(self, nlist):
        self.nlist = nlist
        self.nlist.subscribe(self.get_rcut)
        self.nlist.update_rcut()

    def _initialize_types(self):
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        if self.dimensions == 2:
            for typ in type_list:
                self.setParams2D(typ, [[0, 0]], False)
        else:
            for typ in type_list:
                self.setParams3D(typ, [[0, 0, 0]], [], False)

    def setParams2D(self, type, vertices, center=True):
        """Set the vertices for a given particle type. Takes a type
        name, a list of 2D vertices relative to the center of mass of
        the particle, and a boolean value for whether to center the
        particle automatically."""
        itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(type)

        if not len(vertices):
            vertices = [(0, 0)]
            center = False

        # explicitly turn into a list of tuples
        if center:
            vertices = [(float(p[0]), float(p[1])) for p in utils.center(vertices)]
        else:
            vertices = [(float(p[0]), float(p[1])) for p in vertices]

        # update the neighbor list
        rcutmax = 2*(sqrt(max(x*x + y*y for (x, y) in vertices)) + self.radius*2**(1./6))
        self.r_cut = max(self.r_cut, rcutmax)

        self.vertices[type] = vertices
        self.cpp_force.setRcut(self.r_cut)
        self.cpp_force.setParams(itype, vertices)

    def setParams3D(self, type, vertices, faces, center=True):
        """Set the shape parameters for a given particle type. Takes a
        type name, a list of 3D vertices, a list of lists of vertex
        indices (with one list for each face), and a boolean value for
        whether to center the particle automatically."""
        itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(type)

        if not len(vertices):
            vertices = [(0, 0, 0)]
            faces = []
            center = False

        # explicitly turn into python lists
        if center:
            vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in utils.center(vertices, faces)]
        else:
            vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in vertices]
        faces = [[int(i) for i in face] for face in faces]

        # update the neighbor list
        rcutmax = 2*(sqrt(max(x*x + y*y + z*z for (x, y, z) in vertices)) + self.radius*2**(1./6))
        self.r_cut = max(self.r_cut, rcutmax)

        self.vertices[type] = vertices
        self.cpp_force.setRcut(self.r_cut)
        self.cpp_force.setParams(itype, vertices, faces)

class WCA(hoomd.md.force._force, _DEMBase):
    ## Specify the DEM WCA force
    #
    # \param radius Rounding radius to use for shape
    def __init__(self, nlist, radius=1.):
        hoomd.util.print_status_line();
        friction = None

        self.radius = radius
        self.autotunerEnabled = True
        self.autotunerPeriod = 100000
        self.vertices = {}

        self.onGPU = hoomd.context.exec_conf.isCUDAEnabled()
        cppForces = {(2, None, 'cpu'): _dem.WCADEM2D,
             (2, None, 'gpu'): (_dem.WCADEM2DGPU if self.onGPU else None),
             (3, None, 'cpu'): _dem.WCADEM3D,
             (3, None, 'gpu'): (_dem.WCADEM3DGPU if self.onGPU else None)}

        self.dimensions = hoomd.context.current.system_definition.getNDimensions()

        # initialize the base class
        hoomd.md.force._force.__init__(self);

        # interparticle cutoff radius, will be updated as shapes are added
        self.r_cut = 2*radius*2**(1./6)

        if friction is None:
            potentialParams = params.WCA(radius=radius)
        else:
            raise RuntimeError('Unknown friction type: {}'.format(friction))

        _DEMBase.__init__(self, nlist)

        key = (self.dimensions, friction, 'gpu' if self.onGPU else 'cpu')
        cpp_force = cppForces[key]

        self.cpp_force = cpp_force(hoomd.context.current.system_definition,
                                   self.nlist.cpp_nlist, self.r_cut,
                                   potentialParams)

        if self.dimensions == 2:
            self.setParams = self.setParams2D
        else:
            self.setParams = self.setParams3D

        self._initialize_types()

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def update_coeffs(self):
        """Noop for this potential"""
        pass

    def setAutotunerParams(self, enable=None, period=None):
        if not self.onGPU:
            return
        if enable is not None:
            self.autotunerEnabled = enable
        if period is not None:
            self.autotunerPeriod = period
        self.cpp_force.setAutotunerParams(self.autotunerEnabled, self.autotunerPeriod)

    def get_rcut(self):
        # self.log is True if the force is enabled
        if not self.log:
            return None

        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # update the rcut by pair type
        r_cut_dict = nl.rcut();
        r_max_dict = {typ: sqrt(max(sum(p*p for p in point)
                                    for point in self.vertices[typ]))
                      for typ in self.vertices}
        for i in range(ntypes):
            for j in range(i, ntypes):
                (typei, typej) = type_list[i], type_list[j]
                r_cut_dict.set_pair(typei, typej,
                                    r_max_dict.get(typei, 0) + r_max_dict.get(typej, 0) + self.radius*2*2.0**(1./6))

        r_cut_dict.fill()

        return r_cut_dict

class SWCA(hoomd.md.force._force, _DEMBase):
    ## Specify the DEM WCA force, shifted by particle diameter
    #
    # \param radius Potential lengthscale ( \f$ \sigma = 2*r \f$ )
    def __init__(self, nlist, radius=1., d_max=None):
        hoomd.util.print_status_line();
        friction = None

        self.radius = radius
        self.autotunerEnabled = True
        self.autotunerPeriod = 100000
        self.vertices = {}

        self.onGPU = hoomd.context.exec_conf.isCUDAEnabled()
        cppForces = {(2, None, 'cpu'): _dem.SWCADEM2D,
             (2, None, 'gpu'): (_dem.SWCADEM2DGPU if self.onGPU else None),
             (3, None, 'cpu'): _dem.SWCADEM3D,
             (3, None, 'gpu'): (_dem.SWCADEM3DGPU if self.onGPU else None)}

        self.dimensions = hoomd.context.current.system_definition.getNDimensions()

        # Error out in MPI simulations
        if (hoomd._hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.msg.error("pair.SWCA is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error setting up pair potential.")

        # initialize the base class
        hoomd.md.force._force.__init__(self);

        # update the neighbor list
        if d_max is None :
            sysdef = hoomd.context.current.system_definition;
            self.d_max = max(x.diameter for x in hoomd.data.particle_data(sysdef.getParticleData()))
            hoomd.context.msg.notice(2, "Notice: swca set d_max=" + str(self.d_max) + "\n");

        # interparticle cutoff radius, will be updated as shapes are added
        self.r_cut = 2*2*self.radius*2**(1./6)

        if friction is None:
            potentialParams = params.SWCA(radius=radius)
        else:
            raise RuntimeError('Unknown friction type: {}'.format(friction))

        _DEMBase.__init__(self, nlist)

        key = (self.dimensions, friction, 'gpu' if self.onGPU else 'cpu')
        cpp_force = cppForces[key]

        self.cpp_force = cpp_force(hoomd.context.current.system_definition,
                                   self.nlist.cpp_nlist, self.r_cut,
                                   potentialParams)

        if self.dimensions == 2:
            self.setParams = self.setParams2D
        else:
            self.setParams = self.setParams3D

        self._initialize_types()

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def update_coeffs(self):
        """Noop for this potential"""
        pass

    def setAutotunerParams(self, enable=None, period=None):
        if not self.onGPU:
            return
        if enable is not None:
            self.autotunerEnabled = enable
        if period is not None:
            self.autotunerPeriod = period
        self.cpp_force.setAutotunerParams(self.autotunerEnabled, self.autotunerPeriod)

    def get_rcut(self):
        if not self.log:
            return None

        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # update the rcut by pair type
        r_cut_dict = nl.rcut();
        r_max_dict = {typ: sqrt(max(sum(p*p for p in point)
                                    for point in self.vertices[typ]))
                      for typ in self.vertices}
        for i in range(ntypes):
            for j in range(i, ntypes):
                (typei, typej) = type_list[i], type_list[j]
                r_cut_dict.set_pair(typei, typej,
                                    r_max_dict.get(typei, 0) + r_max_dict.get(typej, 0) +
                                    self.radius*2*2.0**(1./6) + self.d_max - 1)

        r_cut_dict.fill()

        return r_cut_dict
