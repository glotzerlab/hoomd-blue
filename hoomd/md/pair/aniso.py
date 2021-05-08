from hoomd import md
from hoomd.md.pair.pair import Pair
from hoomd.logging import log
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes, OnlyFrom, positive_real


class AnisotropicPair(Pair):
    R"""Generic anisotropic pair potential.

    Users should not instantiate `AnisotropicPair` directly. It is a base
    class that provides common features to all anisotropic pair forces.
    All anisotropic pair potential commands specify that a given potential
    energy, force and torque be computed on all non-excluded particle pairs in
    the system within a short range cutoff distance :math:`r_{\mathrm{cut}}`.
    The interaction energy, forces and torque depend on the inter-particle
    separation :math:`\vec r` and on the orientations :math:`\vec q_i`,
    :math:`q_j`, of the particles.

    `AnisotropicPair` is similiar to `Pair` except it does not support the
    `xplor` shifting mode or `r_on`.

    Args:
        nlist (hoomd.md.nlist.NList) : The neighbor list.
        r_cut (`float`, optional) : The default cutoff for the potential,
            defaults to ``None`` which means no cutoff (units: [length]).
        mode (`str`, optional) : the energy shifting mode, defaults to "none".
    """

    def __init__(self, nlist, r_cut=None, mode="none"):
        self._nlist = OnlyTypes(md.nlist.NList, strict=True)(nlist)
        tp_r_cut = TypeParameter('r_cut', 'particle_types',
                                 TypeParameterDict(positive_real, len_keys=2)
                                 )
        if r_cut is not None:
            tp_r_cut.default = r_cut
        self._param_dict.update(
            ParameterDict(mode=OnlyFrom(['none', 'shift'])))
        self.mode = mode
        self._add_typeparam(tp_r_cut)

    def _return_type_shapes(self):
        type_shapes = self.cpp_force.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret


class Dipole(AnisotropicPair):
    R""" Screened dipole-dipole interactions.

    Implements the force and energy calculations for both magnetic and
    electronic dipole-dipole interactions. When particles have charge as well as
    a dipole moment, the interactions are through electronic dipole moments. If
    the particles have no charge then the interaction is through magnetic or
    electronic dipoles. Note whether a dipole is magnetic or electronic does not
    change the functional form of the potential only the units associated with
    the potential parameters.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        r_cut (float): Default cutoff radius (units: [length]).
        r_on (float): Default turn-on radius (units: [length]).
        mode (str): energy shifting/smoothing mode

    `Dipole` computes the (screened) interaction between pairs of
    particles with dipoles and electrostatic charges. The total energy
    computed is:

    .. math::

        U_{dipole} = U_{dd} + U_{de} + U_{ee}

        U_{dd} = A e^{-\kappa r}
            \left(\frac{\vec{\mu_i}\cdot\vec{\mu_j}}{r^3}
                  - 3\frac{(\vec{\mu_i}\cdot \vec{r_{ji}})
                           (\vec{\mu_j}\cdot \vec{r_{ji}})}
                          {r^5}
            \right)

        U_{de} = A e^{-\kappa r}
            \left(\frac{(\vec{\mu_j}\cdot \vec{r_{ji}})q_i}{r^3}
                - \frac{(\vec{\mu_i}\cdot \vec{r_{ji}})q_j}{r^3}
            \right)

        U_{ee} = A e^{-\kappa r} \frac{q_i q_j}{r}

    See `Pair` for details on how forces are calculated and the
    available energy shifting and smoothing modes.  Use ``params`` dictionary to
    set potential coefficients. The coefficients must be set per unique pair of
    particle types.

    Note:
       All units are given for electronic dipole moments.

    Attributes:
        params (TypeParameter[tuple[``particle_type``, ``particle_type``], dict]):
            The dipole potential parameters. The dictionary has the following
            keys:

            * ``A`` (`float`, **optional**) - :math:`A` - electrostatic energy
              scale (*default*: 1.0) (units: [energy] [length] [charge]^-2)


            * ``kappa`` (`float`, **required**) - :math:`\kappa` - inverse
              screening length (units: [length]^-1)

        mu (TypeParameter[``particle_type``, tuple[float, float, float]):
            :math:`\mu` - the magnetic magnitude of the particle local reference
            frame as a tuple (i.e. :math:`(\mu_x, \mu_y, \mu_z)`) (units:
            [charge] [length]).
    Example::

        nl = nlist.Cell()
        dipole = md.pair.Dipole(nl, r_cut=3.0)
        dipole.params[('A', 'B')] = dict(A=1.0, kappa=4.0)
        dipole.mu[('A', 'B')] = (4.0, 1.0, 0.0)
    """
    _cpp_class_name = "AnisoPotentialPairDipole"

    def __init__(self, nlist, r_cut=None, mode='none'):
        super().__init__(nlist, r_cut, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(A=float, kappa=float, len_keys=2))
        mu = TypeParameter(
            'mu', 'particle_types',
            TypeParameterDict((float, float, float), len_keys=1))
        self._extend_typeparam((params, mu))


class GayBerne(AnisotropicPair):
    R""" Gay-Berne anisotropic pair potential.

    Warning: The code has yet to be updated to the current API.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        r_cut (float): Default cutoff radius (units: [length]).
        r_on (float): Default turn-on radius (units: [length]).
        mode (str): energy shifting/smoothing mode.

    `GayBerne` computes the Gay-Berne potential between anisotropic
    particles.

    This version of the Gay-Berne potential supports identical pairs of uniaxial
    ellipsoids, with orientation-independent energy-well depth.

    The interaction energy for this anisotropic pair potential is
    (`Allen et. al. 2006 <http://dx.doi.org/10.1080/00268970601075238>`_):

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{GB}}(\vec r, \vec e_i, \vec e_j)
            = & 4 \varepsilon \left[ \zeta^{-12} - \zeta^{-6} \right]
            & \zeta < \zeta_{\mathrm{cut}} \\
            = & 0 & \zeta \ge \zeta_{\mathrm{cut}} \\
        \end{eqnarray*}

    .. math::

        \zeta = \left(\frac{r-\sigma+\sigma_{\mathrm{min}}}
                           {\sigma_{\mathrm{min}}}\right)

        \sigma^{-2} = \frac{1}{2} \hat{\vec{r}}
            \cdot \vec{H^{-1}} \cdot \hat{\vec{r}}

        \vec{H} = 2 \ell_\perp^2 \vec{1}
            + (\ell_\parallel^2 - \ell_\perp^2)
              (\vec{e_i} \otimes \vec{e_i} + \vec{e_j} \otimes \vec{e_j})

    with :math:`\sigma_{\mathrm{min}} = 2 \min(\ell_\perp, \ell_\parallel)`.

    The cut-off parameter :math:`r_{\mathrm{cut}}` is defined for two particles
    oriented parallel along the **long** axis, i.e.
    :math:`\zeta_{\mathrm{cut}} = \left(\frac{r-\sigma_{\mathrm{max}}
    + \sigma_{\mathrm{min}}}{\sigma_{\mathrm{min}}}\right)`
    where :math:`\sigma_{\mathrm{max}} = 2 \max(\ell_\perp, \ell_\parallel)` .

    The quantities :math:`\ell_\parallel` and :math:`\ell_\perp` denote the
    semi-axis lengths parallel and perpendicular to particle orientation.

    Use ``params`` dictionary to set potential coefficients. The coefficients
    must be set per unique pair of particle types.

    Attributes:
        params (TypeParameter[tuple[``particle_type``, ``particle_type``], dict]):
            The Gay-Berne potential parameters. The dictionary has the following
            keys:

            * ``epsilon`` (`float`, **required**) - :math:`\varepsilon` (units:
              [energy])

            * ``lperp`` (`float`, **required**) - :math:`\ell_\perp` (units:
              [length])

            * ``lpar`` (`float`, **required**) -  :math:`\ell_\parallel` (units:
              [length])

    Example::

        nl = nlist.Cell()
        gay_berne = md.pair.GayBerne(nlist=nl, r_cut=2.5)
        gay_berne.params[('A', 'A')] = dict(epsilon=1.0, lperp=0.45, lpar=0.5)
        gay_berne.r_cut[('A', 'B')] = 2 ** (1.0 / 6.0)

    """
    _cpp_class_name = "AnisoPotentialPairGB"

    def __init__(self, nlist, r_cut=None, mode='none'):
        super().__init__(nlist, r_cut, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              lperp=float,
                              lpar=float,
                              len_keys=2))
        self._add_typeparam(params)

    @log(category="object")
    def type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:

            >>> gay_berne.type_shapes
            [{'type': 'Ellipsoid', 'a': 1.0, 'b': 1.0, 'c': 1.5}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super()._return_type_shapes()


class alj(ai_pair):
    R"""Anistropic LJ potential.

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list
        name (str): Name of the force instance.
        average_simplices (bool): Whether or not to perform simplex averaging (see below for more details).

    :py:class:`alj` computes the LJ potential between anisotropic particles.
    The anisotropy is implemented as a composite of two interactions, a
    center-center component and a component of interaction measured at the
    closest point of contact between the two particles. The potential supports
    both standard LJ interactions as well as repulsive-only WCA interactions.
    This behavior is controlled using the :code:`alpha` parameter, which can
    take on the following values:

    * :code:`0`:
      All interactions are WCA (no attraction).

    * :code:`1`:
      Center-center interactions include attraction,
      contact-contact interactions are solely repulsive.

    * :code:`2`:
      Center-center interactions are solely repulsive,
      contact-contact interactions include attraction.

    * :code:`3`:
      All interactions include attractive and repulsive components.

    For polytopes, computing interactions using a single contact point leads to
    significant instabilities in the torques because the contact point can jump
    from one end of a face to another in an arbitrarily small time interval. To
    ameliorate this, the alj potential performs a local averaging over all the
    features associated with the closest simplices on two polytopes. This
    averaging can be turned off by setting the ``average_simplices`` argument
    to ``False``.

    Use :py:meth:`pair_coeff.set <coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - *epsilon* - :math:`\varepsilon` (in energy units)
    - *sigma_i* - the insphere radius of the first particle type.
    - *sigma_j* - the insphere radius of the second particle type.
    - *alpha* - Integer 0-3 indicating whether or not to include the attractive
                component of the interaction (see above for details).
    - *contact_sigma_i* - the contact sphere radius of the first type.
      - *optional*: defaults to 0.15*sigma_i
    - *contact_sigma_j* - the contact sphere radius of the second type.
      - *optional*: defaults to 0.15*sigma_j
    - :math:`r_{\mathrm{cut}}` - *r_cut* (in distance units)
      - *optional*: defaults to the global r_cut specified in the pair command

    The following shape parameters may be set per particle type:

    - *vertices* - The vertices of a convex polytope in 2 or 3 dimensions. The
                   array may be :math:`N\times2` or :math:`N\times3` in 2D (in
                   the latter case, the third dimension is ignored).
    - *rounding radii* - The semimajor axes of a rounding ellipsoid. If a
                         single number is specified, the rounding ellipsoid is
                         a sphere.
    - *faces* - The faces of the polyhedron specified as a (possible ragged) 2D
                array of integers. The vertices must be ordered (see
                :meth:`~.convexHull` for more information).

    At least one of ``vertices`` or ``rounding_radii`` must be specified.
    Specifying only ``rounding radii creates an ellipsoid, while specifying
    only vertices creates a convex polytope. In general, the faces will be
    inferred by computing the convex hull of the vertices and merging coplanar
    faces. However, because merging of faces requires applying a numerical
    threshold to find coplanar faces, in some cases the default value may
    result in not all coplanar faces actually being merged. In such cases,
    users can precompute the faces and provide them. The convenience class
    method :meth:`~.convexHull` can be used for this purpose.

    Example::

        nl = nlist.cell()
        alj = pair.alj(r_cut=2.5, nlist=nl)
        alj.pair_coeff.set(
            'A', 'A', epsilon=1.0, sigma_i=2.0, sigma_j=2.0, alpha=0)
    """

    def __init__(self, r_cut, nlist, name=None, average_simplices=True):
        hoomd.util.print_status_line()

        # initialize the base class
        ai_pair.__init__(self, r_cut, nlist, name)

        if not hoomd.context.exec_conf.isCUDAEnabled():
            if hoomd.context.current.system_definition.getNDimensions() == 2:
                cls = _md.AnisoPotentialPairALJ2D
            else:
                cls = _md.AnisoPotentialPairALJ3D
        else:
            self.nlist.cpp_nlist.setStorageMode(
                _md.NeighborList.storageMode.full)
            if hoomd.context.current.system_definition.getNDimensions() == 2:
                cls = _md.AnisoPotentialPairALJ2DGPU
            else:
                cls = _md.AnisoPotentialPairALJ3DGPU

        # create the c++ mirror class
        self.cpp_force = cls(hoomd.context.current.system_definition,
                             self.nlist.cpp_nlist, self.name)
        self.cpp_class = cls

        hoomd.context.current.system.addCompute(
            self.cpp_force, self.force_name)

        # Note that while this is set for the entire pair potential, in
        # practice it is passed through on a per-pair basis as part of the pair
        # params.
        self.average_simplices = average_simplices

        # Setup the coefficent options. Note that the contact sigmas are
        # optional, but if not provided they are computed based on the sigmas
        # so we have to provide dummy values here for checking.
        self.required_coeffs = ['epsilon', 'sigma_i', 'sigma_j', 'alpha',
                                'contact_sigma_i', 'contact_sigma_j']
        self.pair_coeff.set_default_coeff('contact_sigma_i', -1.0);
        self.pair_coeff.set_default_coeff('contact_sigma_j', -1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon']
        sigma_i = coeff['sigma_i']
        sigma_j = coeff['sigma_j']
        alpha = int(coeff['alpha'])

        default_contact_multiplier = 0.15
        contact_sigma_i = coeff['contact_sigma_i']
        contact_sigma_j = coeff['contact_sigma_j']
        if contact_sigma_i == -1:
            contact_sigma_i = sigma_i*default_contact_multiplier
        if contact_sigma_j == -1:
            contact_sigma_j = sigma_j*default_contact_multiplier

        if alpha not in range(4):
            raise ValueError(
                "The alpha parameter must be an integer from 0 to 3.")

        return _md.make_pair_alj_params(
            epsilon, sigma_i, sigma_j, contact_sigma_i, contact_sigma_j, alpha,
            self.average_simplices, hoomd.context.exec_conf)


    ### COPIED FROM dem.utils
    @classmethod
    def convexHull(cls, vertices, tol=1e-6):
        """Compute the 3D convex hull of a set of vertices and merge coplanar faces.

        Args:
            vertices (list): List of (x, y, z) coordinates
            tol (float): Floating point tolerance for merging coplanar faces


        Returns an array of vertices and a list of faces (vertex
        indices) for the convex hull of the given set of vertice.

        .. note::
            This method uses scipy's quickhull wrapper and therefore requires scipy.

        """
        from scipy.spatial import cKDTree, ConvexHull;
        from scipy.sparse.csgraph import connected_components;
        from collections import defaultdict
        import numpy as np

        hull = ConvexHull(vertices);
        # Triangles in the same face will be defined by the same linear equalities
        dist = cKDTree(hull.equations);
        trianglePairs = dist.query_pairs(tol);

        connectivity = np.zeros((len(hull.simplices), len(hull.simplices)), dtype=np.int32);

        for (i, j) in trianglePairs:
            connectivity[i, j] = connectivity[j, i] = 1;

        # connected_components returns (number of faces, cluster index for each input)
        (_, joinTarget) = connected_components(connectivity, directed=False);
        faces = defaultdict(list);
        norms = defaultdict(list);
        for (idx, target) in enumerate(joinTarget):
            faces[target].append(idx);
            norms[target] = hull.equations[idx][:3];

        # a list of sets of all vertex indices in each face
        faceVerts = [set(hull.simplices[faces[faceIndex]].flat) for faceIndex in sorted(faces)];
        # normal vector for each face
        faceNorms = [norms[faceIndex] for faceIndex in sorted(faces)];

        # polygonal faces
        polyFaces = [];
        for (norm, faceIndices) in zip(faceNorms, faceVerts):
            face = np.array(list(faceIndices), dtype=np.uint32);
            N = len(faceIndices);

            r = hull.points[face];
            rcom = np.mean(r, axis=0);

            # plane_{a, b}: basis vectors in the plane
            plane_a = r[0] - rcom;
            plane_a /= np.sqrt(np.sum(plane_a**2));
            plane_b = np.cross(norm, plane_a);

            dr = r - rcom[np.newaxis, :];

            thetas = np.arctan2(dr.dot(plane_b), dr.dot(plane_a));

            sortidx = np.argsort(thetas);

            face = face[sortidx];
            polyFaces.append(face.tolist());

        return (hull.points.tolist(), polyFaces);

    def _set_cpp_shape(self, type_id, type_name):
        # Ensure that shape parameters are always 3D lists, even in 2D.
        # TODO: Ensure that the centroid is contained in the shape.
        # There is always at least one vertex. For ellipsoids, this is just the
        # origin and has no effect.
        import numpy as np

        ndim = hoomd.context.current.system_definition.getNDimensions()

        # Process rounding radius
        rrs = self.shape[type_name].get('rounding_radii', 0)
        try:
            rounding_radii = list(rrs)
            if len(rounding_radii) > 3:
                raise ValueError(
                    "The rounding radius must be a single value or a "
                    "sequence of 1-3 values.")

            ndim = hoomd.context.current.system_definition.getNDimensions()
            if len(rounding_radii) == 1:
                rounding_radii *= ndim
                if ndim == 2:
                    rounding_radii += [0]
            elif ndim == 2:
                if len(rounding_radii) == 2:
                    rounding_radii += [0]
                elif rounding_radii[2] != 0:
                    raise ValueError(
                        "The z dimension rounding radius must be 0 in 2D.")
            elif len(rounding_radii) != 3:
                raise ValueError("Invalid rounding radius in 3D.")
        except TypeError:
            # We were passed a scalar value
            if hoomd.context.current.system_definition.getNDimensions() == 2:
                rounding_radii = [rrs, rrs, 0]
            else:
                rounding_radii = [rrs, rrs, rrs]

        # Process vertices
        vertices = self.shape[type_name].get('vertices')
        if vertices is not None:
            vertices = list(vertices)
            if ndim == 2:
                vertices = [[v[0], v[1], 0] for v in vertices]

            if len(vertices) <= ndim and rrs == 0:
                raise ValueError("Your shape must have at least {} vertices in "
                                "{} dimensions".format(
                                    ndim+1, ndim));

            if np.linalg.norm(np.mean(vertices, axis=0)) > 1e-6:
                raise ValueError(
                    "The vertices must be centered at the centroid of your shape. "
                    "Please subtract the centroid (e.g. via "
                    "`np.mean(vertices, axis=0)`) from the vertices.")

            faces = self.shape[type_name].get('faces')
            if faces is None:
                if ndim == 3:
                    vertices, faces = self.convexHull(vertices)
                else:
                    # The faces don't actually get used for 2D, so just pass a
                    # dummy for now.
                    faces = [[0]]
        else:
            vertices = [[0, 0, 0]]
            faces = [[0]]

        param = _md.make_alj_shape_params(
            vertices, faces, rounding_radii, hoomd.context.exec_conf)
        self.cpp_force.setShape(type_id, param)

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(ai_pair, self)._return_type_shapes()
