# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: jproc

R""" Wall potentials.

Wall potentials add forces to any particles within a certain distance,
:math:`r_{\mathrm{cut}}`, of each wall. In the extrapolated
mode, all particles deemed outside of the wall boundary are included as well.

Wall geometries are used to specify half-spaces. There are two half spaces for
each of the possible geometries included and each can be selected using the
inside parameter. In order to fully specify space, it is necessary that one half
space be closed and one open. Setting *inside=True* for closed half-spaces
and *inside=False* for open ones. See :py:class:`wallpotential` for more
information on how the concept of half-spaces are used in implementing wall forces.

.. attention::
    The current wall force implementation does not support NPT integrators.

Wall groups (:py:class:`group`) are used to pass wall geometries to wall forces.
By themselves, wall groups do nothing. Only when you specify a wall force
(i.e. :py:class:`lj`),  are forces actually applied between the wall and the
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import external
import hoomd;
import math;

#           *** Helpers ***

class group(object):
    R""" Defines a wall group.

    Args:
        walls (list): Wall objects to be included in the group.

    All wall forces use a wall group as an input so it is necessary to create a
    wall group object before any wall force can be created. Modifications
    of the created wall group may occur at any time before :py:func:`hoomd.run`
    is invoked. Current supported geometries are spheres, cylinder, and planes. The
    maximum number of each type of wall is 20, 20, and 60 respectively.

    The **inside** parameter used in each wall geometry is used to specify the
    half-space that is to be used for the force implementation. See
    :py:class:`wallpotential` for more general details and :py:class:`sphere`
    :py:class:`cylinder`, and :py:class:`plane` for the definition of inside for
    each geometry.

    An effective use of wall forces **requires** considering the geometry of the
    system. Walls are only evaluated in one simulation box and are not periodic. It
    is therefore important to ensure that the walls that intersect with a periodic
    boundary meet.

    Note:
        The entire structure can easily be viewed by printing the group object.

    Note:
        While all x,y,z coordinates can be given as a list or tuple, only origin
        parameters are points in x,y,z space. Normal and axis parameters are vectors and
        must have a nonzero magnitude.

    Note:
        Wall structure modifications between :py:func:`hoomd.run`
        calls will be implemented in the next run. However, modifications must be done
        carefully since moving the wall can result in particles moving to a relative
        position which causes exceptionally high forces resulting in particles moving
        many times the box length in one move.

    Example::

        In[0]:
        # Creating wall geometry definitions using convenience functions
        wallstructure=wall.group()
        wallstructure.add_sphere(r=1.0,origin=(0,1,3))
        wallstructure.add_sphere(1.0,[0,-1,3],inside=False)
        wallstructure.add_cylinder(r=1.0,origin=(1,1,1),axis=(0,0,3),inside=True)
        wallstructure.add_cylinder(4.0,[0,0,0],(1,0,1))
        wallstructure.add_cylinder(5.5,(1,1,1),(3,-1,1),False)
        wallstructure.add_plane(origin=(3,2,1),normal=(2,1,4))
        wallstructure.add_plane((0,0,0),(10,2,1))
        wallstructure.add_plane((0,0,0),(0,2,1))
        print(wallstructure)

        Out[0]:
        Wall_Data_Structure:
        spheres:2{
        [0:   Radius=1.0  Origin=(0.0, 1.0, 3.0)  Inside=True]
        [1:   Radius=1.0  Origin=(0.0, -1.0, 3.0) Inside=False]}
        cylinders:3{
        [0:   Radius=1.0  Origin=(1.0, 1.0, 1.0)  Axis=(0.0, 0.0, 3.0)    Inside=True]
        [1:   Radius=4.0  Origin=(0.0, 0.0, 0.0)  Axis=(1.0, 0.0, 1.0)    Inside=True]
        [2:   Radius=5.5  Origin=(1.0, 1.0, 1.0)  Axis=(3.0, -1.0, 1.0)   Inside=False]}
        planes:3{
        [0:   Origin=(3.0, 2.0, 1.0)  Normal=(2.0, 1.0, 4.0)]
        [1:   Origin=(0.0, 0.0, 0.0)  Normal=(10.0, 2.0, 1.0)]
        [2:   Origin=(0.0, 0.0, 0.0)  Normal=(0.0, 2.0, 1.0)]}

        In[1]:
        # Deleting wall geometry definitions using convenience functions in all accepted types
        wallstructure.del_plane(range(3))
        wallstructure.del_cylinder([0,2])
        wallstructure.del_sphere(1)
        print(wallstructure)

        Out[1]:
        Wall_Data_Structure:
        spheres:1{
        [0:   Radius=1.0  Origin=(0.0, 1.0, 3.0)  Inside=True]}
        cylinders:1{
        [0:   Radius=4.0  Origin=(0.0, 0.0, 0.0)  Axis=(1.0, 0.0, 1.0)    Inside=True]}
        planes:0{}

        In[2]:
        # Modifying wall geometry definitions using convenience functions
        wallstructure.spheres[0].r=2.0
        wallstructure.cylinders[0].origin=[1,2,1]
        wallstructure.cylinders[0].axis=(0,0,1)
        print(wallstructure)

        Out[2]:
        Wall_Data_Structure:
        spheres:1{
        [0:   Radius=2.0  Origin=(0.0, 1.0, 3.0)  Inside=True]}
        cylinders:1{
        [0:   Radius=4.0  Origin=(1.0, 2.0, 1.0)  Axis=(0.0, 0.0, 1.0)    Inside=True]}
        planes:0{}

    """
    def __init__(self,*walls):
        self.spheres=[];
        self.cylinders=[];
        self.planes=[];
        for wall in walls:
            self.add(wall);

    def add(self,wall):
        R""" Generic wall add for wall objects.

        Generic convenience function to add any wall object to the group.
        Accepts :py:class:`sphere`, :py:class:`cylinder`, :py:class:`plane`, and lists of any
        combination of these.
        """
        if (isinstance(wall, sphere)):
            self.spheres.append(wall);
        elif (isinstance(wall, cylinder)):
            self.cylinders.append(wall);
        elif (isinstance(wall, plane)):
            self.planes.append(wall);
        elif (type(wall)==list):
            for wall_el in wall:
                if (isinstance(wall_el, sphere)):
                    self.spheres.append(wall_el);
                elif (isinstance(wall_el, cylinder)):
                    self.cylinders.append(wall_el);
                elif (isinstance(wall_el, plane)):
                    self.planes.append(wall_el);
                else:
                    print("Input of type "+str(type(wall_el))+" is not allowed. Skipping invalid list element...");
        else:
            print("Input of type "+str(type(wall))+" is not allowed.");

    def add_sphere(self, r, origin, inside=True):
        R""" Adds a sphere to the wall group.

        Args:
            r (float): Sphere radius (in distance units)
            origin (tuple): Sphere origin (in x,y,z coordinates)
            inside (bool): Selects the half-space to be used (bool)

        Adds a sphere with the specified parameters to the group.
        """
        self.spheres.append(sphere(r,origin,inside));

    def add_cylinder(self, r, origin, axis, inside=True):
        R""" Adds a cylinder to the wall group.

        Args:
            r (float): Cylinder radius (in distance units)
            origin (tuple): Cylinder origin (in x,y,z coordinates)
            axis (tuple): Cylinder axis vector (in x,y,z coordinates)
            inside (bool): Selects the half-space to be used (bool)

        Adds a cylinder with the specified parameters to the group.
        """
        self.cylinders.append(cylinder(r, origin, axis, inside));

    def add_plane(self, origin, normal, inside=True):
        R""" Adds a plane to the wall group.

        Args:
            origin (tuple): Plane origin (in x,y,z coordinates)
            normal (tuple): Plane normal vector (in x,y,z coordinates)
            inside (bool): Selects the half-space to be used (bool)

        Adds a plane with the specified parameters to the wallgroup.planes list.
        """
        self.planes.append(plane(origin, normal, inside));

    def del_sphere(self, *indexs):
        R""" Deletes the sphere or spheres in index.

        Args:
            index (list): The index of sphere(s) desired to delete. Accepts int, range, and lists.

        Removes the specified sphere or spheres from the wallgroup.spheres list.
        """
        for index in indexs:
            if type(index) is int: index = [index];
            elif type(index) is range: index = list(index);
            index=list(set(index));
            index.sort(reverse=True);
            for i in index:
                try:
                    del(self.spheres[i]);
                except IndexValueError:
                    hoomd.context.msg.error("Specified index for deletion is not valid.\n");
                    raise RuntimeError("del_sphere failed")

    def del_cylinder(self, *indexs):
        R""" Deletes the cylinder or cylinders in index.

        Args:
            index (list): The index of cylinder(s) desired to delete. Accepts int, range, and lists.

        Removes the specified cylinder or cylinders from the wallgroup.cylinders list.
        """
        for index in indexs:
            if type(index) is int: index = [index];
            elif type(index) is range: index = list(index);
            index=list(set(index));
            index.sort(reverse=True);
            for i in index:
                try:
                    del(self.cylinders[i]);
                except IndexValueError:
                    hoomd.context.msg.error("Specified index for deletion is not valid.\n");
                    raise RuntimeError("del_cylinder failed")

    def del_plane(self, *indexs):
        R""" Deletes the plane or planes in index.

        Args:
            index (list): The index of plane(s) desired to delete. Accepts int, range, and lists.

        Removes the specified plane or planes from the wallgroup.planes list.
        """
        for index in indexs:
            if type(index) is int: index = [index];
            elif type(index) is range: index = list(index);
            index=list(set(index));
            index.sort(reverse=True);
            for i in index:
                try:
                    del(self.planes[i]);
                except IndexValueError:
                    hoomd.context.msg.error("Specified index for deletion is not valid.\n");
                    raise RuntimeError("del_plane failed")

    ## \internal
    # \brief Return metadata for this wall structure
    def get_metadata(self):
        data = hoomd.meta._metadata_from_dict(eval(str(self.__dict__)));
        return data;

    ## \internal
    # \brief Returns output for print
    def __str__(self):
        output="Wall_Data_Structure:\nspheres:%s{"%(len(self.spheres));
        for index in range(len(self.spheres)):
            output+="\n[%s:\t%s]"%(repr(index), str(self.spheres[index]));

        output+="}\ncylinders:%s{"%(len(self.cylinders));
        for index in range(len(self.cylinders)):
            output+="\n[%s:\t%s]"%(repr(index), str(self.cylinders[index]));

        output+="}\nplanes:%s{"%(len(self.planes));
        for index in range(len(self.planes)):
            output+="\n[%s:\t%s]"%(repr(index), str(self.planes[index]));

        output+="}";
        return output;

class sphere(object):
    R""" Sphere wall.

    Args:
        r (float): Sphere radius (in distance units)
        origin (tuple): Sphere origin (in x,y,z coordinates)
        inside (bool): Selects the half-space to be used

    Define a spherical half-space:

    - inside = True selects the space inside the radius of the sphere and includes the sphere surface.
    - inside = False selects the space outside the radius of the sphere.

    Use in function calls or by reference in the creation or modification of wall groups.

    The following example is intended to demonstrate cylinders and planes
    as well. Note that the distinction between points and vectors is reflected in
    the default parameters.

    Examples::

        In[0]:
        # One line initialization
        one_line_walls=wall.group(wall.sphere(r=3,origin=(0,0,0)),wall.cylinder(r=2.5,axis=(0,0,1),inside=True), wall.plane(normal=(1,0,0)))
        print(one_line_walls)
        full_wall_object=wall.group([wall.sphere()]*20,[wall.cylinder()]*20,[wall.plane()]*60)
        # Sharing wall group elements and access by reference
        common_sphere=wall.sphere()
        linked_walls1=wall.group(common_sphere,wall.plane(origin=(3,0,0),normal=(-1,0,0)))
        linked_walls2=wall.group(common_sphere,wall.plane(origin=(-3,0,0),normal=(1,0,0)))
        common_sphere.r=5.0
        linked_walls1.spheres[0].origin=(0,0,1)
        print(linked_walls1)
        print(linked_walls2)

        Out[0]:
        Wall_Data_Structure:
        spheres:1{
        [0:   Radius=3    Origin=(0.0, 0.0, 0.0)  Inside=True]}
        cylinders:1{
        [0:   Radius=2.5  Origin=(0.0, 0.0, 0.0)  Axis=(0.0, 0.0, 1.0)    Inside=True]}
        planes:1{
        [0:   Origin=(0.0, 0.0, 0.0)  Normal=(1.0, 0.0, 0.0)]}
        Wall_Data_Structure:
        spheres:1{
        [0:   Radius=5.0  Origin=(0.0, 0.0, 1.0)  Inside=True]}
        cylinders:0{}
        planes:1{
        [0:   Origin=(3.0, 0.0, 0.0)  Normal=(-1.0, 0.0, 0.0)]}
        Wall_Data_Structure:
        spheres:1{
        [0:   Radius=5.0  Origin=(0.0, 0.0, 1.0)  Inside=True]}
        cylinders:0{}
        planes:1{
        [0:   Origin=(-3.0, 0.0, 0.0) Normal=(1.0, 0.0, 0.0)]}

    """
    def __init__(self, r=0.0, origin=(0.0, 0.0, 0.0), inside=True):
        self.r = r;
        self._origin = _hoomd.make_scalar3(*origin);
        self.inside = inside;

    @property
    def origin(self):
        return (self._origin.x, self._origin.y, self._origin.z);
    @origin.setter
    def origin(self, origin):
        self._origin = _hoomd.make_scalar3(*origin);

    def __str__(self):
        return "Radius=%s\tOrigin=%s\tInside=%s" % (str(self.r), str(self.origin), str(self.inside));

    def __repr__(self):
        return "{'r': %s, 'origin': %s, 'inside': %s}" % (str(self.r), str(self.origin), str(self.inside));

class cylinder(object):
    R""" Cylinder wall.

    Args:
        r (float): Cylinder radius (in distance units)
        origin (tuple): Cylinder origin (in x,y,z coordinates)
        axis (tuple): Cylinder axis vector (in x,y,z coordinates)
        inside (bool): Selects the half-space to be used (bool)

    Define a cylindrical half-space:

    - inside = True selects the space inside the radius of the cylinder and includes the cylinder surface.
    - inside = False selects the space outside the radius of the cylinder.

    Use in function calls or by reference in the creation or modification of wall groups.

    For an example see :py:class:`sphere`.
    """
    def __init__(self, r=0.0, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), inside=True):
        self.r = r;
        self._origin = _hoomd.make_scalar3(*origin);
        self._axis = _hoomd.make_scalar3(*axis);
        self.inside = inside;

    @property
    def origin(self):
        return (self._origin.x, self._origin.y, self._origin.z);
    @origin.setter
    def origin(self, origin):
        self._origin = _hoomd.make_scalar3(*origin);

    @property
    def axis(self):
        return (self._axis.x, self._axis.y, self._axis.z);
    @axis.setter
    def axis(self, axis):
        self._axis = _hoomd.make_scalar3(*axis);

    def __str__(self):
        return "Radius=%s\tOrigin=%s\tAxis=%s\tInside=%s" % (str(self.r), str(self.origin), str(self.axis), str(self.inside));

    def __repr__(self):
        return "{'r': %s, 'origin': %s, 'axis': %s, 'inside': %s}" % (str(self.r), str(self.origin), str(self.axis), str(self.inside));

class plane(object):
    R""" Plane wall.

    Args:
        origin (tuple): Plane origin (in x,y,z coordinates)\n <i>Default : (0.0, 0.0, 0.0)</i>
        normal (tuple): Plane normal vector (in x,y,z coordinates)\n <i>Default : (0.0, 0.0, 1.0)</i>
        inside (bool): Selects the half-space to be used (bool)\n <i>Default : True</i>

    Define a planar half space.

    - inside = True selects the space on the side of the plane to which the normal vector points and includes the plane surface.
    - inside = False selects the space on the side of the plane opposite the normal vector.

    Use in function calls or by reference in the creation or modification of wall groups.

    For an example see :py:class:`sphere`.
    """
    def __init__(self, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), inside=True):
        self._origin = _hoomd.make_scalar3(*origin);
        self._normal = _hoomd.make_scalar3(*normal);
        self.inside = inside;

    @property
    def origin(self):
        return (self._origin.x, self._origin.y, self._origin.z);
    @origin.setter
    def origin(self, origin):
        self._origin = _hoomd.make_scalar3(*origin);

    @property
    def normal(self):
        return (self._normal.x, self._normal.y, self._normal.z);
    @normal.setter
    def normal(self, normal):
        self._normal = _hoomd.make_scalar3(*normal);

    def __str__(self):
        return "Origin=%s\tNormal=%s\tInside=%s" % (str(self.origin), str(self.normal), str(self.inside));

    def __repr__(self):
        return "{'origin':%s, 'normal': %s, 'inside': %s}" % (str(self.origin), str(self.normal), str(self.inside));

#           *** Potentials ***

class wallpotential(external._external_force):
    R""" Generic wall potential.

    :py:class:`wallpotential` should not be used directly.
    It is a base class that provides common features to all standard wall
    potentials. Rather than repeating all of that documentation in many different
    places, it is collected here.

    All wall potential commands specify that a given potential energy and potential be
    computed on all particles in the system within a cutoff distance,
    :math:`r_{\mathrm{cut}}`, from each wall in the given wall group.
    The force :math:`\vec{F}` is in the direction of :math:`\vec{r}`, the vector
    pointing from the particle to the wall or half-space boundary and
    :math:`V_{\mathrm{pair}}(r)` is the specific pair potential chosen by the
    respective command. Wall forces are implemented with the concept of half-spaces
    in mind. There are two modes which are allowed currently in wall potentials:
    standard and extrapolated.

    .. rubric:: Standard Mode.

    In the standard mode, when :math:`r_{\mathrm{extrap}} \le 0`, the potential
    energy is only applied to the half-space specified in the wall group. :math:`V(r)`
    is evaluated in the same manner as when the mode is shift for the analogous :py:mod:`pair <hoomd.md.pair>`
    potentials within the boundaries of the half-space.

    .. math::

        V(r)  = V_{\mathrm{pair}}(r) - V_{\mathrm{pair}}(r_{\mathrm{cut}})

    For inside=True (closed) half-spaces:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}  = & -\nabla V(r) & 0 \le r < r_{\mathrm{cut}} \\
                 = & 0 & r \ge r_{\mathrm{cut}} \\
                 = & 0 & r < 0
        \end{eqnarray*}

    For inside=False (open) half-spaces:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}  = & -\nabla V(r) & 0 < r < r_{\mathrm{cut}} \\
                 = & 0 & r \ge r_{\mathrm{cut}} \\
                 = & 0 & r \le 0
        \end{eqnarray*}

    .. rubric: Extrapolated Mode:

    The wall potential can be linearly extrapolated beyond a minimum separation from the wall
    :math:`r_{\mathrm{extrap}}` in the active half-space. This can be useful for bringing particles outside the
    half-space into the active half-space. It can also be useful for typical wall force usages by
    effectively limiting the maximum force experienced by the particle due to the wall. The potential is extrapolated into **both**
    half-spaces and the cutoff :math:`r_{\mathrm{cut}}` only applies in the active half-space. The user should
    then be careful using this mode with multiple nested walls. It is intended to be used primarily for initialization.

    The extrapolated potential has the following form:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{extrap}}(r) =& V(r) &, r > r_{\rm extrap} \\
             =& V(r_{\rm extrap}) + (r_{\rm extrap}-r)\vec{F}(r_{\rm extrap}) \cdot \vec{n}&, r \le r_{\rm extrap}
        \end{eqnarray*}

    where :math:`\vec{n}` is the normal into the active half-space.
    This gives an effective force on the particle due to the wall:

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}(r) =& \vec{F}_{\rm pair}(r) &, r > r_{\rm extrap} \\
                =& \vec{F}_{\rm pair}(r_{\rm extrap}) &, r \le r_{\rm extrap}
        \end{eqnarray*}

    where :math:`\vec{F}_{\rm pair}` is given by the gradient of the pair force

    .. math::
        :nowrap:

        \begin{eqnarray*}
        \vec{F}_{\rm pair}(r) =& -\nabla V_{\rm pair}(r) &, r < r_{\rm cut} \\
                           =& 0 &, r \ge r_{\mathrm{cut}}
        \end{eqnarray*}

    In other words, if :math:`r_{\rm extrap}` is chosen so that the pair force would point into the active half-space,
    the extrapolated potential will push all particles into the active half-space. See :py:class:`lj` for a
    pictorial example.

    To use extrapolated mode, the following coefficients must be set per unique particle types:

    - All parameters required by the pair potential base for the wall potential
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units) - *Optional: Defaults to global r_cut for the force if given or 0.0 if not*
    - :math:`r_{\mathrm{extrap}}` - *r_extrap* (in distance units) - *Optional: Defaults to 0.0*


    .. rubric:: Generic Example

    Note that the walls object below must be created before it is given as an
    argument to the force object. However, walls can be modified at any time before
    :py:func:`hoomd.run()` is called and it will update itself appropriately. See
    :py:class:`group` for more details about specifying the walls to be used::

        walls=wall.group()
        # Edit walls
        my_force=wall.pairpotential(walls)
        my_force.force_coeff.set('A', all required arguments)
        my_force.force_coeff.set(['B','C'],r_cut=0.3, all required arguments)
        my_force.force_coeff.set(['B','C'],r_extrap=0.3, all required arguments)

    A specific example can be found in :py:class:`lj`

    .. attention::
        The current wall force implementation does not support NPT integrators.

    Note:
        The virial due to walls is computed, but the pressure and reported by :py:class:`hoomd.analyze.log`
        is not well defined. The volume (area) of the box enters into the pressure computation, which is
        not correct in a confined system. It may not even be possible to define an appropriate volume with
        soft walls.

    Note:
        An effective use of wall forces **requires** considering the geometry of the
        system. Each wall is only evaluated in one simulation box and thus is not
        periodic. Forces will be evaluated and added to all particles from all walls in
        the wall group. Additionally there are no safeguards
        requiring a wall to exist inside the box to have interactions. This means that
        an attractive force existing outside the simulation box would pull particles
        across the periodic boundary where they would immediately cease to have any
        interaction with that wall. It is therefore up to the user to use walls in a
        physically meaningful manner. This includes the geometry of the walls, their
        interactions, and as noted here their location.

    Note:
        When :math:`r_{\mathrm{cut}} \le 0` or is set to False the particle type
        wall interaction is excluded.

    Note:
        While wall potentials are based on the same potential energy calculations
        as pair potentials, Features of pair potentials such as specified neighborlists,
        and alternative force shifting modes are not supported.
    """
    def __init__(self, walls, r_cut, name=""):
        external._external_force.__init__(self, name);
        self.field_coeff = walls;
        self.required_coeffs = ["r_cut", "r_extrap"];
        self.force_coeff.set_default_coeff('r_extrap', 0.0);

        # convert r_cut False to a floating point type
        if (r_cut==False):
            r_cut = 0.0
        self.global_r_cut = r_cut;
        self.force_coeff.set_default_coeff('r_cut', self.global_r_cut);

    ## \internal
    # \brief passes the wall field
    def process_field_coeff(self, coeff):
        return _md.make_wall_field_params(coeff, hoomd.context.exec_conf);

    ## \internal
    # \brief Return metadata for this wall potential
    def get_metadata(self):
        data=external._external_force.get_metadata(self);
        data['walls_struct'] = self.field_coeff.get_metadata();
        return data

    ## \internal
    # \brief Fixes negative values to zero before squaring
    def update_coeffs(self):
        if not self.force_coeff.verify(self.required_coeffs):
            raise RuntimeError('Error updating force coefficients')

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        for i in range(0,ntypes):
            type=hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            if self.force_coeff.values[str(type)]['r_cut']<=0:
                self.force_coeff.values[str(type)]['r_cut']=0;
        external._external_force.update_coeffs(self);


class lj(wallpotential):
    R""" Lennard-Jones wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the Lennard-Jones potential.
    See :py:class:`hoomd.md.pair.lj` for force details and base parameters and
    :py:class:`wallpotential` for generalized wall potential implementation

    Standard mode::

        walls=wall.group()
        #add walls
        lj=wall.lj(walls, r_cut=3.0)
        lj.force_coeff.set('A', sigma=1.0,epsilon=1.0)  #plotted below in red
        lj.force_coeff.set('B', sigma=1.0,epsilon=1.0, r_cut=2.0**(1.0/2.0))
        lj.force_coeff.set(['A','B'], epsilon=2.0, sigma=1.0, alpha=1.0, r_cut=3.0)

    Extrapolated mode::

        walls=wall.group()
        #add walls
        lj_extrap=wall.lj(walls, r_cut=3.0)
        lj_extrap.force_coeff.set('A', sigma=1.0,epsilon=1.0, r_extrap=1.1) #plotted in blue below

    V(r) plot:

    .. image:: wall_extrap.png
    """
    def __init__(self, walls, r_cut=False, name=""):
        hoomd.util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialLJ(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialLJ;
        else:

            self.cpp_force = _md.WallsPotentialLJGPU(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialLJGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha'];
        self.force_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return _md.make_wall_lj_params(_hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

class gauss(wallpotential):
    R""" Gaussian wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the Gaussian potential.
    See :py:class:`hoomd.md.pair.gauss` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_gauss=wall.gauss(walls, r_cut=3.0)
        wall_force_gauss.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        wall_force_gauss.force_coeff.set('A', epsilon=2.0, sigma=1.0, r_cut=3.0)
        wall_force_gauss.force_coeff.set(['C', 'D'], epsilon=3.0, sigma=0.5)

    """
    def __init__(self, walls, r_cut=False, name=""):
        hoomd.util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);
        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialGauss(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialGauss;
        else:

            self.cpp_force = _md.WallsPotentialGaussGPU(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialGaussGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma'];

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        return _md.make_wall_gauss_params(_hoomd.make_scalar2(epsilon, sigma), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

class slj(wallpotential):
    R""" Shifted Lennard-Jones wall potential

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the Shifted Lennard-Jones potential.
    Note that because slj is dependent upon particle diameters the following
    correction is necessary to the force details in the :py:class:`hoomd.md.pair.slj` description.

    :math:`\Delta = d_i/2 - 1` where :math:`d_i` is the diameter of particle :math:`i`.
    See :py:class:`hoomd.md.pair.slj` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_slj=wall.slj(walls, r_cut=3.0)
        wall_force_slj.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        wall_force_slj.force_coeff.set('A', epsilon=2.0, sigma=1.0, r_cut=3.0)
        wall_force_slj.force_coeff.set('B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0))

    """
    def __init__(self, walls, r_cut=False, d_max=None, name=""):
        hoomd.util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # update the neighbor list
        if d_max is None :
            sysdef = hoomd.context.current.system_definition;
            d_max = sysdef.getParticleData().getMaxDiameter()
            hoomd.context.msg.notice(2, "Notice: slj set d_max=" + str(d_max) + "\n");

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialSLJ(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialSLJ;
        else:

            self.cpp_force = _md.WallsPotentialSLJGPU(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialSLJGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha'];
        self.force_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return _md.make_wall_slj_params(_hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

class yukawa(wallpotential):
    R""" Yukawa wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the Yukawa potential.
    See :py:class:`hoomd.md.pair.yukawa` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_yukawa=wall.yukawa(walls, r_cut=3.0)
        wall_force_yukawa.force_coeff.set('A', epsilon=1.0, kappa=1.0)
        wall_force_yukawa.force_coeff.set('A', epsilon=2.0, kappa=0.5, r_cut=3.0)
        wall_force_yukawa.force_coeff.set(['C', 'D'], epsilon=0.5, kappa=3.0)

    """
    def __init__(self, walls, r_cut=False, name=""):
        hoomd.util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialYukawa(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialYukawa;
        else:
            self.cpp_force = _md.WallsPotentialYukawaGPU(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialYukawaGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'kappa'];

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        kappa = coeff['kappa'];
        return _md.make_wall_yukawa_params(_hoomd.make_scalar2(epsilon, kappa), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

class morse(wallpotential):
    R""" Morse wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the Morse potential.
    See :py:class:`hoomd.md.pair.morse` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_morse=wall.morse(walls, r_cut=3.0)
        wall_force_morse.force_coeff.set('A', D0=1.0, alpha=3.0, r0=1.0)
        wall_force_morse.force_coeff.set('A', D0=1.0, alpha=3.0, r0=1.0, r_cut=3.0)
        wall_force_morse.force_coeff.set(['C', 'D'], D0=1.0, alpha=3.0)

    """
    def __init__(self, walls, r_cut=False, name=""):
        hoomd.util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialMorse(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialMorse;
        else:

            self.cpp_force = _md.WallsPotentialMorseGPU(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialMorseGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs += ['D0', 'alpha', 'r0'];

    def process_coeff(self, coeff):
        D0 = coeff['D0'];
        alpha = coeff['alpha'];
        r0 = coeff['r0']

        return _md.make_wall_morse_params(_hoomd.make_scalar4(D0, alpha, r0, 0.0), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

class force_shifted_lj(wallpotential):
    R""" Force-shifted Lennard-Jones wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the Force-shifted Lennard-Jones potential.
    See :py:class:`hoomd.md.pair.force_shifted_lj` for force details and base parameters and :py:class:`wallpotential`
    for generalized wall potential implementation.

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_fslj=wall.force_shifted_lj(walls, r_cut=3.0)
        wall_force_fslj.force_coeff.set('A', epsilon=1.0, sigma=1.0)
        wall_force_fslj.force_coeff.set('B', epsilon=1.5, sigma=3.0, r_cut = 8.0)
        wall_force_fslj.force_coeff.set(['C','D'], epsilon=1.0, sigma=1.0, alpha = 1.5)

    """
    def __init__(self, walls, r_cut=False, name=""):
        hoomd.util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialForceShiftedLJ(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialForceShiftedLJ;
        else:

            self.cpp_force = _md.WallsPotentialForceShiftedLJGPU(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialForceShiftedLJGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha'];
        self.force_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return _md.make_wall_force_shift_lj_params(_hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

class mie(wallpotential):
    R""" Mie potential wall potential.

    Args:
        walls (:py:class:`group`): Wall group containing half-space geometries for the force to act in.
        r_cut (float): The global r_cut value for the force. Defaults to False or 0 if not specified.
        name (str): The force name which will be used in the metadata and log files.

    Wall force evaluated using the Mie potential.
    See :py:class:`hoomd.md.pair.mie` for force details and base parameters and :py:class:`wallpotential` for
    generalized wall potential implementation

    Example::

        walls=wall.group()
        # add walls to interact with
        wall_force_mie=wall.mie(walls, r_cut=3.0)
        wall_force_mie.force_coeff.set('A', epsilon=1.0, sigma=1.0, n=12, m=6)
        wall_force_mie.force_coeff.set('A', epsilon=2.0, sigma=1.0, n=14, m=7, r_cut=3.0)
        wall_force_mie.force_coeff.set('B', epsilon=1.0, sigma=1.0, n=15.1, m=6.5, r_cut=2**(1.0/6.0))

    """
    def __init__(self, walls, r_cut=False, name=""):
        hoomd.util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.WallsPotentialMie(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialMie;
        else:

            self.cpp_force = _md.WallsPotentialMieGPU(hoomd.context.current.system_definition, self.name);
            self.cpp_class = _md.WallsPotentialMieGPU;

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'n', 'm'];

    def process_coeff(self, coeff):
        epsilon = float(coeff['epsilon']);
        sigma = float(coeff['sigma']);
        n = float(coeff['n']);
        m = float(coeff['m']);

        mie1 = epsilon * math.pow(sigma, n) * (n/(n-m)) * math.pow(n/m,m/(n-m));
        mie2 = epsilon * math.pow(sigma, m) * (n/(n-m)) * math.pow(n/m,m/(n-m));
        mie3 = n
        mie4 = m
        return _md.make_wall_mie_params(_hoomd.make_scalar4(mie1, mie2, mie3, mie4), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);
