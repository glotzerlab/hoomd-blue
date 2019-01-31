# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" HPMC utilities
"""

from __future__ import print_function
from __future__ import division
# If numpy is unavailable, some utilities will not work
try:
    import numpy as np
except ImportError:
    np = None

import hoomd
import sys
import colorsys as cs
import re

#replace range with xrange for python3 compatibility
if sys.version_info[0]==2:
    range=xrange

# Multiply two quaternions
# Apply quaternion multiplication per http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# (requires numpy)
# \param q1 quaternion
# \param q2 quaternion
# \returns q1*q2
def quatMult(q1, q2):
    s = q1[0]
    v = q1[1:]
    t = q2[0]
    w = q2[1:]
    q = np.empty((4,), dtype=float)
    q[0] = s*t - np.dot(v, w)
    q[1:] = s*w + t*v + np.cross(v,w)
    return q

# Rotate a vector by a unit quaternion
# Quaternion rotation per http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
# (requires numpy)
# \param q rotation quaternion
# \param v 3d vector to be rotated
# \returns q*v*q^{-1}
def quatRot(q, v):
    v = np.asarray(v)
    q = np.asarray(q)
    # assume q is a unit quaternion
    w = q[0]
    r = q[1:]
    vnew = np.empty((3,), dtype=v.dtype)
    vnew = v + 2*np.cross(r, np.cross(r,v) + w*v)
    return vnew

# Construct a box matrix from a hoomd data.boxdim object
# (requires numpy)
# \param box hoomd boxdim object
# \returns numpy matrix that transforms lattice coordinates to Cartesian coordinates
def matFromBox(box):
    Lx, Ly, Lz = box.Lx, box.Ly, box.Lz
    xy = box.xy
    xz = box.xz
    yz = box.yz
    return np.matrix([[Lx, xy*Ly, xz*Lz], [0, Ly, yz*Lz], [0, 0, Lz]])

# Given a set of lattice vectors, rotate to produce an upper triangular right-handed box
# as a hoomd boxdim object and a rotation quaternion that brings particles in the original coordinate system to the new one.
# The conversion preserves handedness, so it is left to the user to provide a right-handed set of lattice vectors
# (E.g. returns (data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.5, yz=0.1), q) )
# (requires numpy)
# \param a1 first lattice vector
# \param a2 second lattice vector
# \param a3 third lattice vector
# \returns (box, q) tuple of boxdim object and rotation quaternion
def latticeToHoomd(a1, a2, a3=[0.,0.,1.], ndim=3):
    from hoomd.data import boxdim

    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)
    a1.resize((3,))
    a2.resize((3,))
    a3.resize((3,))
    xhat = np.array([1.,0.,0.])
    yhat = np.array([0.,1.,0.])
    zhat = np.array([0.,0.,1.])

    # Find quaternion to rotate first lattice vector to x axis
    a1mag = np.sqrt(np.dot(a1,a1))
    v1 = a1/a1mag + xhat
    v1mag = np.sqrt(np.dot(v1,v1))
    if v1mag > 1e-6:
        u1 = v1/v1mag
    else:
        # a1 is antialigned with xhat, so rotate around any unit vector perpendicular to xhat
        u1 = yhat
    q1 = np.concatenate(([np.cos(np.pi/2)], np.sin(np.pi/2)*u1))

    # Find quaternion to rotate second lattice vector to xy plane after applying above rotation
    a2prime = quatRot(q1, a2)
    angle = -1*np.arctan2(a2prime[2], a2prime[1])
    q2 = np.concatenate(([np.cos(angle/2)], np.sin(angle/2)*xhat))

    q = quatMult(q2,q1)

    Lx = np.sqrt(np.dot(a1, a1))
    a2x = np.dot(a1, a2) / Lx
    Ly = np.sqrt(np.dot(a2,a2) - a2x*a2x)
    xy = a2x / Ly
    v0xv1 = np.cross(a1, a2)
    v0xv1mag = np.sqrt(np.dot(v0xv1, v0xv1))
    Lz = np.dot(a3, v0xv1) / v0xv1mag
    a3x = np.dot(a1, a3) / Lx
    xz = a3x / Lz
    yz = (np.dot(a2, a3) - a2x*a3x) / (Ly*Lz)

    box = boxdim(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz, dimensions=ndim)

    return box, q

## Read a single (final) frame pos file and return some data structures.
# Returns particle positions, orientations, types, definitions, and a hoomd.data.boxdim object,
# along with the rotation quaternion that rotates the input to output boxes.
# Note that the orientations array will have (meaningless) quaternions for spheres
# If you need to read multiple frames, consider the POS trajectory reader in Freud.
# (requires numpy)
# \param fname input file name
# \param ndim number of dimensions (default 3) for boxdim object
# \returns dict with keys positions,orientations,types,param_dict,box,q
def read_pos(fname, ndim=3):
    from hoomd.data import boxdim
    Lx=0; Ly=0; Lz=0; xy=0; xz=0; yz=0;

    f_in = open(fname)

    positions=[] # list of particle positions
    orientations=[] # list of particle orientations
    types=[] # list of particle types

    t_rs={} # dictionary of position lists for defined particle types
    t_qs={} # dictionary of orientation lists for defined particle types
    t_params={} # additional particle parameters

    eof = 0 # count the number of frames
    for line in f_in:
        # get tokens on line
        tokens = line.split()
        if len(tokens)>2: # definition line
            if re.match('^def', line) is not None:
                # get particle type definition
                t = tokens[1]
                # add new definitions
                t_rs[t]=[]
                t_qs[t]=[]
                t_params[t]={}

                tokens[2]=tokens[2].split('"')[-1]
                tokens[-1]=tokens[-1].split('"')[0]
                shape = tokens[2]
                t_params[t]['shape'] = shape
                if shape == "poly3d":
                    num_verts=int(tokens[3])
                    t_params[t]['verts']=np.array([float(n) for n in tokens[4:num_verts*3+4]]).reshape(num_verts,3)
                if shape == "spoly3d":
                    radius = float(tokens[3])
                    t_params[t]['radius'] = radius
                    num_verts = int(tokens[4])
                    t_params[t]['verts']=np.array([float(n) for n in tokens[5:num_verts*3+5]]).reshape(num_verts,3)
                if shape == "sphere":
                    diameter = float(tokens[3])
                    t_params[t]['diameter'] = diameter
                if shape == "cyl":
                    diameter = float(tokens[3])
                    t_params[t]['diameter'] = diameter
                    t_params[t]['length'] = float(tokens[4])
        if re.match('^box ', line) is not None:
            Lx=float(tokens[-3])
            Ly=float(tokens[-2])
            Lz=float(tokens[-1])
            box = boxdim(Lx, Ly, Lz)
            q = np.array([1,0,0,0])
        elif re.match('^boxMatrix',line) is not None:
            # rotate the box matrix to be upper triangular
            pbox=np.array([float(f) for f in tokens[1:10]]).reshape(3,3)
            a1 = pbox[:,0]
            a2 = pbox[:,1]
            a3 = pbox[:,2]
            box, q = latticeToHoomd(a1, a2, a3, ndim)
        elif re.match('^eof',line) is not None: # clean up at the end of the frame
            eof += 1
            positions = np.concatenate([t_rs[t] for t in t_rs])
            orientations = np.concatenate([t_qs[t] for t in t_rs])
            types = []
            for t in t_rs:
                # build list of particle types with indices corresponding to other data structures
                types.extend([t]*len(t_rs[t]))

            # reset data structures for next frame
            t_rs = {}
            t_qs = {}
            # leave t_params intact... seems like the right thing to do
        else:
            for t in t_rs.keys():
                if re.match('^{}'.format(t), line) is not None:
                    # spheres don't have orientations in pos files
                    if len(tokens) > 4:
                        t_rs[t].append([float(f) for f in tokens[-7:-4]])
                        t_qs[t].append([float(f) for f in tokens[-4:]])
                    else:
                        t_rs[t].append([float(f) for f in tokens[-3:]])
                        t_qs[t].append([1.,0,0,0])

    # Try to recover from missing 'eof' string in single-frame files
    if eof == 0:
        positions = np.concatenate([t_rs[t] for t in t_rs])
        orientations = np.concatenate([t_qs[t] for t in t_rs])
        types = []
        for t in t_rs:
            # build list of particle types with indices corresponding to other data structures
            types.extend([t]*len(t_rs[t]))
        if len(positions) == 0:
            raise ValueError("No frames read from {}. Invalid pos file?".format(fname))

    # rotate particles and positions, then wrap back into box just in case
    for i in range(len(positions)):
        positions[i] = quatRot(q, positions[i])
        positions[i], img = box.wrap(positions[i])
        orientations[i] = quatMult(q, orientations[i])
    f_in.close()
    return {'positions':positions, 'orientations':orientations, 'types':types, 'param_dict':t_params, 'box':box, 'q':q}

## Given an HPMC NPT system as input, perform compression to search for dense packings
# This class conveniently encapsulates the scripting and heuristics to search
# for densest packings.
# The read_pos() module may also be of use.
# (requires numpy)
class compress:
    ## Construct a hpmc.util.compress object.
    # Attach to a hoomd hpmc integrator instance with an existing hpmc.update.npt object.
    # \param mc hpmc integrator object
    # \param npt_updater hpmc.update.npt object
    # \param ptypes list of particle type names
    # \param pnums list of number of particles of each type
    # \param pvolumes list of particle volumes for each type
    # \param pverts list of sets of vertices for each particle type (set empty list for spheres, etc)
    # \param num_comp_steps number of steps over which to ramps up pressure (default 1e6)
    # \param refine_steps number of steps between checking eta at high pressure (default num_comp_steps/10)
    # \param log_file file in which to log hpmc stuff
    # \param pmin low pressure end of pressure schedule (default 10)
    # \param pmax high pressure end of pressure schedule (default 1e6)
    # \param pf_tol tolerance allowed in checking for convergence
    # \param allowShearing allow box to shear when searching for dense packing (default True)
    # \param tuner_period interval sufficient to get statistics on at least ~10,000 particle overlaps and ~100 box changes (default 1000)
    # \param relax number of steps to run at initial box size at each sweep before starting pressure schedule (default 10,000)
    # \param quiet suppress the noisier aspects of hoomd during compression (default True)
    def __init__(self,
                 mc=None,
                 npt_updater=None,
                 ptypes=None,
                 pnums=None,
                 pvolumes=None,
                 pverts=None,
                 num_comp_steps=int(1e6),
                 refine_steps=None,
                 log_file="densest_packing.txt",
                 pmin=10,
                 pmax=1e6,
                 pf_tol=0,
                 allowShearing=True,
                 tuner_period=10000,
                 relax=int(1e4),
                 quiet=True,
                 ):

        # Gather and check arguments
        self.mc = mc
        if self.mc is None:
            raise TypeError("Needs mc.")
        self.npt_updater = npt_updater
        if self.npt_updater is None:
            raise TypeError("Needs npt_updater.")
        self.pnums = pnums
        if self.pnums is None:
            raise TypeError("Needs pnums.")
        self.pvolumes = pvolumes
        if self.pvolumes is None:
            raise TypeError("Needs pvolumes.")
        self.num_comp_steps = int(num_comp_steps)
        self.refine_steps = refine_steps
        if self.refine_steps is None:
            self.refine_steps = self.num_comp_steps // 10
        self.log_file = str(log_file)
        self.pmin = float(pmin)
        self.pmax = float(pmax)
        self.pf_tol = float(pf_tol)
        self.allowShearing=bool(allowShearing)
        self.ptypes = ptypes
        if self.ptypes is None:
            raise TypeError("Needs ptypes")
        self.pverts = pverts
        if self.pverts is None:
            raise TypeError("Needs pverts")
        self.tuner_period = int(tuner_period)
        self.relax = int(relax)
        self.quiet = bool(quiet)

        # Gather additional necessary data
        system = hoomd.data.system_data(hoomd.context.current.system_definition)
        self.dim = system.sysdef.getNDimensions()
        self.tot_pvol = np.dot(self.pnums, self.pvolumes)
        self.eta_list = list()
        self.snap_list = list()
        box = system.sysdef.getParticleData().getGlobalBox()
        Lx = box.getL().x
        Ly = box.getL().y
        Lz = box.getL().z
        xy = box.getTiltFactorXY()
        xz = box.getTiltFactorXZ()
        yz = box.getTiltFactorYZ()
        self.box_params = (Lx, Ly, Lz, xy, xz, yz)

        #
        # set up tuners (with some sanity checking)
        #

        self.tuners = list()
        particle_tunables = list()
        max_part_moves = list()
        #don't tune translation with only one particle
        if system.sysdef.getParticleData().getN() > 1:
            particle_tunables.append('d')
            max_part_moves.append(0.25)
        #if all particles are spheres, don't tune rotation
        if any([len(vert)>1 and n>0 for n,vert in zip(self.pnums,self.pverts)]):
            particle_tunables.append('a')
            max_part_moves.append(0.5)
        if len(particle_tunables) > 0:
            particle_tuner = tune(obj=self.mc, tunables=particle_tunables, max_val=max_part_moves, gamma=0.3)
            self.tuners.append(particle_tuner)

        box_tunables = ['dLx', 'dLy']
        if self.dim==3:
            box_tunables += ['dLz']
        if self.allowShearing:
            box_tunables += ['dxy']
            if self.dim==3:
                box_tunables += ['dxz', 'dyz']

        box_tuner = tune_npt(obj=self.npt_updater, tunables=box_tunables, gamma=1.0)
        self.tuners.append(box_tuner)

        #
        #Set up the logger
        #

        log_values = [
                    'hpmc_boxmc_betaP',
                    'volume',
                    'hpmc_d',
                    'hpmc_a',
                    'hpmc_boxmc_volume_acceptance',
                    'hpmc_boxmc_shear_acceptance',
                    ]
        self.mclog = hoomd.analyze.log(filename=self.log_file, quantities=log_values , period=self.tuner_period, header_prefix='#', overwrite=True)
        self.mclog.disable() # will be enabled and disabled by call to run()

    ## Run one or more compression cycles
    # \param num_comp_cycles number of compression cycles to run (default 1)
    # \returns tuple of lists of packing fractions and corresponding snapshot objects.
    def run(self, num_comp_cycles=1):
        ## construct exponentially growing pressure variant
        # \param num_comp_steps number of steps in pressure variant
        # \param pmin minimum pressure
        # \param pmax maximum pressure
        # \returns P pressure variant for use in NPT updater
        def makePvariant(num_comp_steps, pmin, pmax):
            num_points = 101 # number of points defining the curve
            interval = num_comp_steps / num_points
            pressures=np.logspace(np.log10(pmin), np.log10(pmax), num_points)
            P = hoomd.variant.linear_interp(points = [(i*interval, prs) for i,prs in enumerate(pressures)])
            return P

        num_comp_cycles = int(num_comp_cycles)

        dim = self.dim
        pmin = self.pmin
        pmax = self.pmax
        allowShearing = self.allowShearing
        num_comp_steps = self.num_comp_steps
        tot_pvol = self.tot_pvol
        (Lx, Ly, Lz, xy, xz, yz) = self.box_params
        relax = self.relax
        refine_steps = self.refine_steps
        quiet = self.quiet
        tuner_period = self.tuner_period
        log_file = self.log_file
        ptypes = self.ptypes
        pf_tol = self.pf_tol

        self.mclog.enable()
        # Since a logger will output on the current step and then every period steps, we need to take one step
        # to get the logger in sync with our for loop.
        hoomd.run(1, quiet=True)

        #
        # set up NPT npt_updater
        #

        Lscale = 0.001
        Ascale = A3scale = 0.01
        if (dim==2):
            A3scale=0.0

        self.npt_updater.set_betap(pmin)
        self.npt_updater.length(delta=Lscale)
        if allowShearing:
            self.npt_updater.shear(delta=A3scale, reduce=0.6)

        #calculate initial packing fraction
        volume = Lx*Ly if dim==2 else Lx*Ly*Lz
        last_eta = tot_pvol / volume
        hoomd.context.msg.notice(5,'Starting eta = {}. '.format(last_eta))
        hoomd.context.msg.notice(5,'Starting volume = {}. '.format(volume))
        hoomd.context.msg.notice(5,'overlaps={}.\n'.format(self.mc.count_overlaps()))

        for i in range(num_comp_cycles):
            hoomd.context.msg.notice(5,'Compressor sweep {}. '.format(i))

            # if not first sweep, relax the system
            if i != 0:
                # set box volume to original
                hoomd.update.box_resize(Lx = Lx, Ly = Ly, Lz = Lz, period=None)
                # reset tunables
                self.npt_updater.set_betap(pmin)
                self.npt_updater.length(delta=Lscale)
                if allowShearing:
                    self.npt_updater.shear(delta=A3scale)
                self.mc.set_params(d=0.1, a=0.01)

            noverlaps = self.mc.count_overlaps()
            if noverlaps != 0:
                hoomd.util.quiet_status()
                hoomd.context.msg.warning("Tuner cannot run properly if overlaps exist in the system. Expanding box...\n")
                while noverlaps != 0:
                    hoomd.context.msg.notice(5,"{} overlaps at step {}... ".format(noverlaps, hoomd.get_step()))
                    Lx *= 1.0+Lscale
                    Ly *= 1.0+Lscale
                    Lz *= 1.0+Lscale
                    hoomd.update.box_resize(Lx = Lx, Ly = Ly, Lz = Lz, period=None)
                    noverlaps = self.mc.count_overlaps()
                hoomd.util.unquiet_status()

            #randomize the initial configuration
            #initial box, no shear
            pretuning_steps = relax
            hoomd.run(pretuning_steps, quiet=quiet)

            # update pressure variant
            P = makePvariant(num_comp_steps, pmin, pmax)
            self.npt_updater.set_betap(P)

            # determine number of iterations for tuner loops
            loop_length = 0
            for tuner in self.tuners:
                loop_length += int(tuner_period)
            #num_iterations = (num_comp_steps) // loop_length
            num_iterations = (num_comp_steps - pretuning_steps) // loop_length

            # run short loops with tuners until pressure is maxed out
            for j in range(num_iterations):
                for tuner in self.tuners:
                    hoomd.run(tuner_period, quiet=quiet)
                    tuner.update()

            #calculate packing fraction for zeroth iteration
            hoomd.context.msg.notice(5,"Checking eta at step {0}. ".format(hoomd.get_step()))
            L = hoomd.context.current.system_definition.getParticleData().getGlobalBox().getL()
            volume = L.x * L.y if dim==2 else L.x*L.y*L.z
            eta = tot_pvol / volume
            hoomd.context.msg.notice(5,'eta = {}, '.format(eta))
            hoomd.context.msg.notice(5,"volume: {0}\n".format(volume))

            step = hoomd.get_step()
            last_step = step
            j = 0
            max_eta_checks = 100
            # If packing has not converged, iterate until it does. Run at least one iteration
            last_eta = 0.0
            while (eta - last_eta) > pf_tol:
                hoomd.run(refine_steps, quiet=quiet)
                # check eta
                hoomd.context.msg.notice(5,"Checking eta at step {0}. ".format(hoomd.get_step()))
                #calculate the new packing fraction
                L = hoomd.context.current.system_definition.getParticleData().getGlobalBox().getL()
                volume = L.x * L.y if dim==2 else L.x*L.y*L.z
                last_eta = eta
                eta = tot_pvol / volume
                hoomd.context.msg.notice(5,"eta: {0}, ".format(eta))
                hoomd.context.msg.notice(5,"volume: {0}\n".format(volume))
                last_step = step
                step = hoomd.get_step()
                # Check if we've gone too far
                if j == max_eta_checks:
                    hoomd.context.msg.notice(5,"Eta did not converge in {0} iterations. Continuing to next cycle anyway.\n".format(max_eta_checks))
                j += 1

            hoomd.context.msg.notice(5,"Step: {step}, Packing fraction: {eta}, ".format(step=last_step, eta=last_eta))
            hoomd.context.msg.notice(5,'overlaps={}\n'.format(self.mc.count_overlaps()))
            self.eta_list.append(last_eta)

            #take a snapshot of the system
            snap = snapshot()
            self.mc.setup_pos_writer(snap)
            self.snap_list.append(snap)

        self.mclog.disable()
        return (self.eta_list,self.snap_list)


## snapshot is a python struct for now, will eventually be replaced with by the hoomd snapshot
# For now, this will be used by the compressor. snapshots can be written to file to_pos method
# In order to write out, the snapshot must be given particle data via the integrator's
# setup_pos_writer() method or all particles will be output as spheres.
# (requires numpy)
#
# \par Quick Example
# \code
# system = init.initmethod(...);
# mc = hpmc.integrate.shape(...);
# mc.shape_param[name].set(...);
# run(...);
# mysnap = hpmc.util.snapshot();
# mc.setup_pos_writer(mysnap, colors=dict(A='ff5984ff'));
# mysnap.to_pos(filename);
# \endcode
class snapshot:
    ## constructor
    def __init__(self):
        system = hoomd.data.system_data(hoomd.context.current.system_definition)
        box = system.sysdef.getParticleData().getGlobalBox()
        L = box.getL()
        xy = box.getTiltFactorXY()
        xz = box.getTiltFactorXZ()
        yz = box.getTiltFactorYZ()
        self.dim = system.sysdef.getNDimensions()
        self.Lx = L.x
        self.Ly = L.y
        self.Lz = L.z
        self.xy = xy
        self.xz = xz
        self.yz = yz
        self.positions = np.array([p.position for p in system.particles])
        self.orientations = np.array([p.orientation for p in system.particles])
        self.ptypes = [p.type for p in system.particles]
        self.ntypes = system.sysdef.getParticleData().getNTypes()
        self.type_list = []
        for i in range(0,self.ntypes):
            self.type_list.append(system.sysdef.getParticleData().getNameByType(i));

        # set up default shape definitions in case set_def is not called
        self.tdef = dict()
        colors=[ 'ff'+''.join([str(c) for c in (256*np.array(cs.hsv_to_rgb(h,0.7,0.7)))]) for h in np.linspace(0.0,0.8,self.ntypes)]
        for i in range(len(self.type_list)):
            t = self.type_list[i]
            # to avoid injavis errors due to presence of orientations, use spoly3d instead of sphere
            self.tdef[t] = 'spoly3d 1 1 0 0 0 {}'.format(colors[i])

    ## \internal Set up particle type definition strings for pos file output
    # This method is intended only to be called by an integrator instance as a result of
    # a call to the integrator's mc.setup_pos_writer() method.
    # \param ptype particle type name (string)
    # \param shapedef pos file particle macro for shape parameters through color
    # \returns None
    def set_def(self, ptype, shapedef):
        self.tdef[ptype] = shapedef

    ## write to a pos file
    # \param filename string name of output file in injavis/incsim pos format
    # \returns None
    def to_pos(self,filename):
        ofile = open(filename, 'w')
        Lx, Ly, Lz, xy, xz, yz = self.Lx, self.Ly, self.Lz, self.xy, self.xz, self.yz
        bmatrix = [Lx, Ly*xy, Lz*xz, 0.0, Ly, Lz*yz, 0.0, 0.0, Lz]
        bmstring = ('boxMatrix ' + 9*'{} ' + '\n').format(*bmatrix)
        ofile.write(bmstring)

        for p in self.tdef:
            ofile.write('def ' + p + ' "' + self.tdef[p] + '"\n')

        for r,q,t in zip(self.positions, self.orientations, self.ptypes):
            outline = t + (3*' {}').format(*r) + (4*' {}').format(*q) + '\n'
            ofile.write(outline)

        ofile.write('eof\n')
        ofile.close()

    ## write to a zip file
    # Not yet implemented
    # \param filename string name of output file in injavis/incsim pos format
    # \returns None
    def to_zip(self,filename):
        raise NotImplementedError("snapshot.to_zip not yet implemented.")

class tune(object):
    R""" Tune mc parameters.

    ``hoomd.hpmc.util.tune`` provides a general tool to observe Monte Carlo move
    acceptance rates and adjust the move sizes when called by a user script. By
    default, it understands how to read and adjust the trial move domain for
    translation moves and rotation moves for an ``hpmc.integrate`` instance.
    Other move types for integrators or updaters can be handled with a customized
    tunable map passed when creating the tuner or in a subclass definition. E.g.
    see use an implementation of :py:class:`.tune_npt`

    Args:
        obj: HPMC Integrator or Updater instance
        tunables (list): list of strings naming parameters to tune. By default,
            allowed element values are 'd' and 'a'.
        max_val (list): maximum allowed values for corresponding tunables
        target (float): desired acceptance rate
        max_scale (float): maximum amount to scale a parameter in a single update
        gamma (float): damping factor (>= 0.0) to keep from scaling parameter values too fast
        type (str): Name of a single hoomd particle type for which to tune move sizes.
            If None (default), all types are tuned with the same statistics.
        tunable_map (dict): For each tunable, provide a dictionary of values and methods to be used by the tuner (see below)
        args: Additional positional arguments
        kwargs: Additional keyword arguments

    Example::

        mc = hpmc.integrate.convex_polyhedron()
        mc.set_params(d=0.01, a=0.01, move_ratio=0.5)
        tuner = hpmc.util.tune(mc, tunables=['d', 'a'], target=0.2, gamma=0.5)
        for i in range(10):
            run(1e4)
            tuner.update()

    Note:
        You should run enough steps to get good statistics for the acceptance ratios. 10,000 trial moves
        seems like a good number. E.g. for 10,000 or more particles, tuning after a single timestep should be fine.
        For npt moves made once per timestep, tuning as frequently as 1,000 timesteps could get a rough convergence
        of acceptance ratios, which is probably good enough since we don't really know the optimal acceptance ratio, anyway.

    Warning:
        There are some sanity checks that are not performed. For example, you shouldn't try to scale 'd' in a single particle simulation.

    Details:

    If ``gamma == 0``, each call to :py:meth:`.update` rescales the current
    value of the tunable\(s\) by the ratio of the observed acceptance rate to the
    target value. For ``gamma > 0``, the scale factor is the reciprocal of
    a weighted mean of the above ratio with 1, according to

        scale = (1.0 + gamma) / (target/acceptance + gamma)

    The names in ``tunables`` must match one of the keys in ``tunable_map``,
    which in turn correspond to the keyword parameters of the MC object being
    updated.

    ``tunable_map`` is a :py:class:`dict` of :py:class:`dict`. The keys of the
    outer :py:class:`dict` are strings that can be specified in the ``tunables``
    parameter. The value of this outer :py:class:`dict` is another :py:class:`dict`
    with the following four keys: 'get', 'acceptance', 'set', and 'maximum'.

    A default ``tunable_map`` is provided but can be modified or extended by setting
    the following dictionary key/value pairs in the entry for tunable.

    * get (:py:obj:`callable`): function called by tuner (no arguments) to retrieve current tunable value
    * acceptance (:py:obj:`callable`): function called by tuner (no arguments) to get relevant acceptance rate
    * set (:py:obj:`callable`): function to call to set new value (optional). Must take one argument (the new value).
      If not provided, ``obj.set_params(tunable=x)`` will be called to set the new value.
    * maximum (:py:class:`float`): maximum value the tuner may set for the tunable parameter

    The default ``tunable_map`` defines the :py:obj:`callable` for 'set' to call
    :py:meth:`hoomd.hpmc.integrate.mode_hpmc.set_params` with ``tunable={type: newval}``
    instead of ``tunable=newval`` if the ``type`` argument is given when creating
    the ``tune`` object.

    """
    def __init__(self, obj=None, tunables=[], max_val=[], target=0.2, max_scale=2.0, gamma=2.0, type=None, tunable_map=None, *args, **kwargs):
        hoomd.util.quiet_status()

        # The *args and **kwargs parameters allow derived tuners to be sloppy
        # with forwarding initialization, but that is not a good excuse and
        # makes it harder to catch usage errors. They should probably be deprecated...

        # Ensure that max_val conforms to the tunable list provided
        max_val_length = len(max_val)
        if (max_val_length != 0) and (max_val_length != len(tunables)):
            raise ValueError("If provided, max_val must be same length as tunables.")

        # Use the default tunable map if none is provided
        if (tunable_map is None):
            tunable_map = dict()
            if type is None:
                tunable_map.update({'d': {
                                                     'get': lambda: getattr(obj, 'get_d')(),
                                                     'acceptance': lambda: getattr(obj, 'get_translate_acceptance')(),
                                                     'set': lambda x: getattr(obj, 'set_params')(d=x),
                                                     'maximum': 1.0
                                                      },
                                          'a': {
                                                     'get': lambda: getattr(obj, 'get_a')(),
                                                     'acceptance': lambda: getattr(obj, 'get_rotate_acceptance')(),
                                                     'set': lambda x: getattr(obj, 'set_params')(a=x),
                                                     'maximum': 0.5
                                                      }})
            else:
                tunable_map.update({'d': {
                                                 'get': lambda: getattr(obj, 'get_d')(type),
                                                 'acceptance': lambda: getattr(obj, 'get_translate_acceptance')(),
                                                 'set': lambda x: getattr(obj, 'set_params')(d={type: x}),
                                                 'maximum': 1.0
                                                 },
                                              'a': {
                                                 'get': lambda: getattr(obj, 'get_a')(type),
                                                 'acceptance': lambda: getattr(obj, 'get_rotate_acceptance')(),
                                                 'set': lambda x: getattr(obj, 'set_params')(a={type: x}),
                                                 'maximum': 0.5
                                                 }})

        #init rest of tuner
        self.target = float(target)
        self.max_scale = float(max_scale)
        self.gamma = float(gamma)
        allowed_tunables = set(tunable_map.keys())
        # list of maximum values for active tunables; not used internally
        self.maxima = list()
        # mapping of tunables and acceptance ratios being tracked to information and methods
        self.tunables = dict()

        # map tunable parameters to a tuple defining
        #  (0) lambda expression to retrieve current value
        #  (1) lambda expression to retrieve acceptance rate
        #  (2) sensible maximum allowed value

        for i in range(len(tunables)):
            item = tunables[i]
            if item in allowed_tunables:
                self.tunables[item] = tunable_map[item]
                if max_val_length != 0:
                    self.tunables[item]['maximum'] = max_val[i]
                self.maxima.append(self.tunables[item]['maximum'])
            else:
                raise ValueError( "Unknown tunable {0}".format(item))

        hoomd.util.unquiet_status()

    def update(self):
        R""" Calculate and set tunable parameters using statistics from the run just completed.
        """
        hoomd.util.quiet_status()

        # Note: we are not doing any checking on the quality of our retrieved statistics
        newquantities = dict()
        # For each of the tunables we are watching, compute the new value we're setting that tunable to
        for tunable in self.tunables:
            oldval = self.tunables[tunable]['get']()
            acceptance = self.tunables[tunable]['acceptance']()
            max_val = self.tunables[tunable]['maximum']
            if (acceptance > 0.0):
                # find (damped) scale somewhere between 1.0 and acceptance/target
                scale = ((1.0 + self.gamma) * acceptance) / (self.target + self.gamma * acceptance)
            else:
                # acceptance rate was zero. Try a parameter value an order of magnitude smaller
                scale = 0.1
            if (scale > self.max_scale):
                scale = self.max_scale
            # find new value
            if (oldval == 0):
                newval = 1e-5
                hoomd.context.msg.warning("Oops. Somehow {0} went to zero at previous update. Resetting to {1}.\n".format(tunable, newval))
            else:
                newval = float(scale * oldval)
                # perform sanity checking on newval
                if (newval == 0.0):
                    newval = float(1e-6)
                if (newval > max_val):
                    newval = max_val

            self.tunables[tunable]['set'](float(newval))
        hoomd.util.unquiet_status();

class tune_npt(tune):
    R""" Tune the HPMC :py:class:`hoomd.hpmc.update.boxmc` using :py:class:`.tune`.

    This is a thin wrapper to ``tune`` that simply defines an alternative
    ``tunable_map`` dictionary. In this case, the ``obj`` argument must be an instance of
    :py:class:`hoomd.hpmc.update.boxmc`. Several tunables are defined.

    'dLx', 'dLy', and 'dLz' use the acceptance rate of volume moves to set
    ``delta[0]``, ``delta[1]``, and ``delta[2]``, respectively in a call to :py:meth:`hoomd.hpmc.update.boxmc.length`.

    'dV' uses the volume acceptance to call :py:meth:`hoomd.hpmc.update.boxmc.volume`.

    'dlnV' uses the ln_volume acceptance to call :py:meth:`hoomd.hpmc.update.boxmc.ln_volume`.

    'dxy', 'dxz', and 'dyz' tunables use the shear acceptance to set
    ``delta[0]``, ``delta[1]``, and ``delta[2]``, respectively in a call to
    :py:meth:`hoomd.hpmc.update.boxmc.shear`.

    Refer to the documentation for :py:class:`hoomd.hpmc.update.boxmc` for
    information on how these parameters are used, since they are not all
    applicable for a given use of ``boxmc``.

    Note:
        A bigger damping factor gamma may be appropriate for tuning box volume
        changes because there may be multiple parameters affecting each acceptance rate.

    Example::

        mc = hpmc.integrate.convex_polyhedron()
        mc.set_params(d=0.01, a=0.01, move_ratio=0.5)
        updater = hpmc.update.boxmc(mc, betaP=10)
        updater.length(0.1, weight=1)
        tuner = hpmc.util.tune_npt(updater, tunables=['dLx', 'dLy', 'dLz'], target=0.3, gamma=1.0)
        for i in range(10):
            run(1e4)
            tuner.update()

    """
    def __init__(self, obj=None, tunables=[], max_val=[], target=0.2, max_scale=2.0, gamma=2.0, type=None, tunable_map=None, *args, **kwargs):
        hoomd.util.quiet_status()
        tunable_map = {
                    'dLx': {
                          'get': lambda: obj.length()['delta'][0],
                          'acceptance': obj.get_volume_acceptance,
                          'maximum': 1.0,
                          'set': lambda x: obj.length(delta=(x, obj.length()['delta'][1], obj.length()['delta'][2]))
                          },
                    'dLy': {
                          'get': lambda: obj.length()['delta'][1],
                          'acceptance': obj.get_volume_acceptance,
                          'maximum': 1.0,
                          'set': lambda x: obj.length(delta=(obj.length()['delta'][0], x, obj.length()['delta'][2]))
                          },
                    'dLz': {
                          'get': lambda: obj.length()['delta'][2],
                          'acceptance': obj.get_volume_acceptance,
                          'maximum': 1.0,
                          'set': lambda x: obj.length(delta=(obj.length()['delta'][0], obj.length()['delta'][1], x))
                          },
                    'dV': {
                          'get': lambda: obj.volume()['delta'],
                          'acceptance': obj.get_volume_acceptance,
                          'maximum': 1.0,
                          'set': lambda x: obj.volume(delta=x)
                          },
                    'dlnV': {
                          'get': lambda: obj.ln_volume()['delta'],
                          'acceptance': obj.get_ln_volume_acceptance,
                          'maximum': 1.0,
                          'set': lambda x: obj.ln_volume(delta=x)
                          },
                    'dxy': {
                          'get': lambda: obj.shear()['delta'][0],
                          'acceptance': obj.get_shear_acceptance,
                          'maximum': 1.0,
                          'set': lambda x: obj.shear(delta=(x, obj.shear()['delta'][1], obj.shear()['delta'][2]))
                          },
                    'dxz': {
                          'get': lambda: obj.shear()['delta'][1],
                          'acceptance': obj.get_shear_acceptance,
                          'maximum': 1.0,
                          'set': lambda x: obj.shear(delta=(obj.shear()['delta'][0], x, obj.shear()['delta'][2]))
                          },
                    'dyz': {
                          'get': lambda: obj.shear()['delta'][2],
                          'acceptance': obj.get_shear_acceptance,
                          'maximum': 1.0,
                          'set': lambda x: obj.shear(delta=(obj.shear()['delta'][0], obj.shear()['delta'][1], x))
                          },
                    }
        hoomd.util.unquiet_status()
        super(tune_npt,self).__init__(obj, tunables, max_val, target, max_scale, gamma, type, tunable_map, *args, **kwargs)
