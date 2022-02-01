# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Anisotropic potentials."""

from collections.abc import Sequence
import json
from numbers import Number

from hoomd import md
from hoomd.md.pair.pair import Pair
from hoomd.logging import log
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import (OnlyTypes, OnlyFrom, positive_real,
                                      OnlyIf, to_type_converter)


class AnisotropicPair(Pair):
    r"""Generic anisotropic pair potential.

    Users should not instantiate `AnisotropicPair` directly. It is a base
    class that provides common features to all anisotropic pair forces.
    All anisotropic pair potential commands specify that a given potential
    energy, force and torque be computed on all non-excluded particle pairs in
    the system within a short range cutoff distance :math:`r_{\mathrm{cut}}`.
    The interaction energy, forces and torque depend on the inter-particle
    separation :math:`\vec r` and on the orientations :math:`\vec q_i`,
    :math:`q_j`, of the particles.

    `AnisotropicPair` is similar to `hoomd.md.pair.Pair` except it does not
    support the ``'xplor'`` shifting mode or `r_on`.

    Args:
        nlist (hoomd.md.nlist.NList) : The neighbor list.
        default_r_cut (`float`, optional) : The default cutoff for the
            potential, defaults to ``None`` which means no cutoff
            :math:`[\mathrm{length}]`.
        mode (`str`, optional) : the energy shifting mode, defaults to "none".
    """

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        self._nlist = OnlyTypes(md.nlist.NList, strict=True)(nlist)
        tp_r_cut = TypeParameter('r_cut', 'particle_types',
                                 TypeParameterDict(positive_real, len_keys=2))
        if default_r_cut is not None:
            tp_r_cut.default = default_r_cut
        self._param_dict.update(ParameterDict(mode=OnlyFrom(['none', 'shift'])))
        self.mode = mode
        self._add_typeparam(tp_r_cut)

    def _return_type_shapes(self):
        type_shapes = self._cpp_obj.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret


class Dipole(AnisotropicPair):
    r"""Screened dipole-dipole interactions.

    Implements the force and energy calculations for both magnetic and
    electronic dipole-dipole interactions. When particles have charge as well as
    a dipole moment, the interactions are through electronic dipole moments. If
    the particles have no charge then the interaction is through magnetic or
    electronic dipoles. Note whether a dipole is magnetic or electronic does not
    change the functional form of the potential only the units associated with
    the potential parameters.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
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

    See `hoomd.md.pair.Pair` for details on how forces are calculated and the
    available energy shifting and smoothing modes.  Use ``params`` dictionary to
    set potential coefficients. The coefficients must be set per unique pair of
    particle types.

    Note:
       All units are given for electronic dipole moments.

    .. py:attribute:: params

        The dipole potential parameters. The dictionary has the following
        keys:

        * ``A`` (`float`, **required**) - :math:`A` - electrostatic energy
          scale (*default*: 1.0)
          :math:`[\mathrm{energy} \cdot \mathrm{length} \cdot
          \mathrm{charge}^{-2}]`
        * ``kappa`` (`float`, **required**) - :math:`\kappa` - inverse
          screening length :math:`[\mathrm{length}^{-1}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    .. py:attribute:: mu

        :math:`\mu` - the magnetic magnitude of the particle local reference
        frame as a tuple (i.e. :math:`(\mu_x, \mu_y, \mu_z)`)
        :math:`[\mathrm{charge} \cdot \mathrm{length}]`.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float` ]]

    Example::

        nl = nlist.Cell()
        dipole = md.pair.Dipole(nl, default_r_cut=3.0)
        dipole.params[('A', 'B')] = dict(A=1.0, kappa=4.0)
        dipole.mu['A'] = (4.0, 1.0, 0.0)
    """
    _cpp_class_name = "AnisoPotentialPairDipole"

    def __init__(self, nlist, default_r_cut=None, mode='none'):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(A=float, kappa=float, len_keys=2))
        mu = TypeParameter('mu', 'particle_types',
                           TypeParameterDict((float, float, float), len_keys=1))
        self._extend_typeparam((params, mu))


class GayBerne(AnisotropicPair):
    r"""Gay-Berne anisotropic pair potential.

    Warning: The code has yet to be updated to the current API.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): energy shifting/smoothing mode.

    `GayBerne` computes the Gay-Berne potential between anisotropic
    particles.

    This version of the Gay-Berne potential supports identical pairs of uniaxial
    ellipsoids, with orientation-independent energy-well depth. The potential
    comes from the following paper Allen et. al. 2006 `paper link`_.

    .. _paper link: http://dx.doi.org/10.1080/00268970601075238

    The interaction energy for this anisotropic pair potential is

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{GB}}(\vec r, \vec e_i, \vec e_j)
            = & 4 \varepsilon \left[ \zeta^{-12} - \zeta^{-6} \right];
            & \zeta < \zeta_{\mathrm{cut}} \\
            = & 0; & \zeta \ge \zeta_{\mathrm{cut}} \\
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

    .. py:attribute:: params

        The Gay-Berne potential parameters. The dictionary has the following
        keys:

        * ``epsilon`` (`float`, **required**) - :math:`\varepsilon`
          :math:`[\mathrm{energy}]`
        * ``lperp`` (`float`, **required**) - :math:`\ell_\perp`
          :math:`[\mathrm{length}]`
        * ``lpar`` (`float`, **required**) -  :math:`\ell_\parallel`
          :math:`[\mathrm{length}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]


    Example::

        nl = nlist.Cell()
        gay_berne = md.pair.GayBerne(nlist=nl, default_r_cut=2.5)
        gay_berne.params[('A', 'A')] = dict(epsilon=1.0, lperp=0.45, lpar=0.5)
        gay_berne.r_cut[('A', 'B')] = 2 ** (1.0 / 6.0)

    """
    _cpp_class_name = "AnisoPotentialPairGB"

    def __init__(self, nlist, default_r_cut=None, mode='none'):
        super().__init__(nlist, default_r_cut, mode)
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


class ALJ(AnisotropicPair):
    r"""Anistropic LJ potential.

    Args:
        nlist (hoomd.md.nlist.NList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[length]`.
        mode (`str`, optional) : the energy shifting mode, defaults to "none".

    `ALJ` computes the Lennard-Jones potential between anisotropic particles as
    described in `Ramasubramani, V.  et. al. 2020`_. Specifically we implement
    the formula:

    .. _Ramasubramani, V.  et. al. 2020: https://doi.org/10.1063/5.0019735

    .. math::
        :nowrap:

        \begin{eqnarray*}
        V_{\mathrm{ALJ}}(r, r_c)
            = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
            \left( \frac{\sigma}{r} \right)^{6} \right] +
            4 \varepsilon_c \left[ \left( \frac{\sigma_c}{r_c} \right)^{12} -
            \left( \frac{\sigma_c}{r_c} \right)^{6} \right] \\
        \end{eqnarray*}

    The first term is the standard center-center interaction between two
    Lennard-Jones spheres. The second term is a contact interaction computed
    based on the smallest distance between the surfaces of the two shapes,
    :math:`r_c`. The total potential energy can thus be viewed as the sum of
    two interactions, a central Lennard-Jones potential and a shifted
    Lennard-Jones potential where the shift is anisotroipc and depends on the
    extent of the shape in each direction.

    Like a standard LJ potential, each term has an independent cutoff beyond
    which it decays to zero the behavior of these cutoffs is dependendent on
    whether a user requires LJ or Weeks-Chandler-Anderson (WCA)-like
    (repulsive-only) behavior. This behavior is controlled using the ``alpha``
    parameter, which can take on the following values:

    * 0:
      All interactions are WCA (no attraction).

    * 1:
      Center-center interactions include attraction,
      contact-contact interactions are solely repulsive.

    * 2:
      Center-center interactions are solely repulsive,
      contact-contact interactions include attraction.

    * 3:
      All interactions include attractive and repulsive components.

    For polytopes, computing interactions using a single contact point leads to
    significant instabilities in the torques because the contact point can jump
    from one end of a face to another in an arbitrarily small time interval. To
    ameliorate this, the ALJ potential performs a local averaging over all the
    features associated with the closest simplices on two polytopes. This
    averaging can be turned off by setting the ``average_simplices`` key for the
    type pair to ``False``.

    .. py:attribute:: params

        The ALJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - base energy scale
          :math:`\varepsilon` :math:`[energy]`.
        * ``sigma_i`` (`float`, **required**) - the insphere radius of the first
          particle type, :math:`[length]`.
        * ``sigma_j`` (`float`, **required**) - the insphere radius of the
          second particle type, :math:`[length]`.
        * ``alpha`` (`int`, **required**) - Integer 0-3 indicating whether or
          not to include the attractive component of the interaction (see
          above for details).
        * ``contact_ratio_i`` (`float`, **optional**) - the ratio of the contact
          sphere radius of the first type with ``sigma_i``. Defaults to 0.15.
        * ``contact_ratio_j`` (`float`, **optional**) - the ratio of the contact
          sphere radius of the second type with ``sigma_j``. Defaults to 0.15.
        * ``average_simplices`` (`bool`, **optional**) - Whether to average over
          simplices. Defaults to ``True``. See class documentation for more
          information.

        Type: `hoomd.data.typeparam.TypeParameter` [`tuple` [``particle_types``,
        ``particle_types``], `dict`]

    .. py:attribute:: shape

        The shape of a given type. The dictionary has the following keys per
        type:

        * ``vertices`` (`list` [`tuple` [`float`, `float`, `float`]],
          **required**) - The vertices of a convex polytope in 2 or 3
          dimensions. The third dimension in 2D is ignored.
        * ``rounding_radii`` (`tuple` [`float`, `float`, `float`] or `float`,
          **required**) - The semimajor axes of a rounding ellipsoid. If a
          single value is specified, the rounding ellipsoid is a sphere.
        * ``faces`` (`list` [`list` [`int`]], **required**) - The faces of the
          polyhedron specified as a list of list of integers.  The vertices
          must be ordered (see `get_ordered_vertices` for more information).

        Type: `hoomd.data.typeparam.TypeParameter` [``particle_types``, `dict`]

    Specifying only ``rounding_radii`` creates an ellipsoid, while specifying
    only ``vertices`` creates a convex polytope (set ``vertices`` and ``faces``
    to empty list to create the ellipsoid). To automate the computation of
    faces, the convenience class method `get_ordered_vertices` can be used.
    However, because merging of faces requires applying a numerical threshold to
    find coplanar faces, in some cases `get_ordered_vertices` may result in not
    all coplanar faces actually being merged. In such cases, users can
    precompute the faces and provide them.

    Example::

        nl = hoomd.md.nlist.Cell()
        alj = hoomd.md.pair.aniso.ALJ(nl, r_cut=2.5)

        cube_verts = [(-0.5, -0.5, -0.5),
                      (-0.5, -0.5, 0.5),
                      (-0.5, 0.5, -0.5),
                      (-0.5, 0.5, 0.5),
                      (0.5, -0.5, -0.5),
                      (0.5, -0.5, 0.5),
                      (0.5, 0.5, -0.5),
                      (0.5, 0.5, 0.5)];

        cube_faces = [[0, 2, 6],
                      [6, 4, 0],
                      [5, 0, 4],
                      [5,1,0],
                      [5,4,6],
                      [5,6,7],
                      [3,2,0],
                      [3,0,1],
                      [3,6,2],
                      [3,7,6],
                      [3,1,5],
                      [3,5,7]]

        alj.params[("A", "A")] = dict(epsilon=2.0,
                                      sigma_i=1.0,
                                      sigma_j=1.0,
                                      alpha=1,
                                      )
        alj.shape["A"] = dict(vertices=cube_verts,
                              faces=cube_faces,
                              rounding_radii=1)

    Warning:
        Changing dimension in a simulation will invalidate this force and will
        lead to error or unrealistic behavior.
    """

    # We don't define a _cpp_class_name since the dimension is a template
    # parameter in C++, so use an instance level attribute instead that is
    # created in _attach based on the dimension of the associated simulation.

    def __init__(self, nlist, default_r_cut=None, mode='none'):
        super().__init__(nlist, default_r_cut, mode)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(epsilon=float,
                              sigma_i=float,
                              sigma_j=float,
                              alpha=int,
                              contact_ratio_i=0.15,
                              contact_ratio_j=0.15,
                              average_simplices=True,
                              len_keys=2))

        shape = TypeParameter(
            'shape', 'particle_types',
            TypeParameterDict(vertices=[(float, float, float)],
                              faces=[[int]],
                              rounding_radii=OnlyIf(
                                  to_type_converter((float, float, float)),
                                  preprocess=self._to_three_tuple),
                              len_keys=1))

        self._extend_typeparam((params, shape))

    @staticmethod
    def get_ordered_vertices(vertices, return_faces=True):
        """Compute vertices and faces of a convex hull of given vertices.

        Warning:
            This method requires the `coxeter <https://coxeter.readthedocs.io/>`_ package.

        Args:
            vertices (:math:`(N_v, 3)` numpy.ndarray of float): The vertices to
                take the convex hull of and get ordered vertices and faces from.
            return_faces (`bool`, optional): Whether to return faces as a list
                of list of int which index into the returned vertices. Defaults
                to ``True``. If ``False`` only vertices are returned and the
                return type is not a tuple.

        Returns:
            tuple: A tuple containing:

                * ``vertices`` (:math:`(N_v, 3)` `numpy.ndarray` of `float`) -
                  The vertices of the convex hull.
                * ``faces`` (`list` [`list` [`int`]]) - The indices into the
                  vertices of vertices defining the faces.
        """
        try:
            import coxeter
        except ImportError as error:
            raise RuntimeError(
                "Method requires coxeter as a dependency.") from error

        shape = coxeter.shapes.ConvexPolyhedron(vertices)

        if return_faces:
            return shape.vertices, shape.faces
        else:
            return shape.vertices

    def _attach(self):
        self._cpp_class_name = "AnisoPotentialPairALJ{}".format(
            "2D" if self._simulation.state.box.is2D else "3D")

        super()._attach()

    @staticmethod
    def _to_three_tuple(value):
        if isinstance(value, Sequence):
            return value
        if isinstance(value, Number):
            return (value, value, value)
        else:
            raise ValueError(f"Expected a float or tuple object got {value}")

    @log(category="object", requires_run=True)
    def type_shapes(self):
        """`list` [`dict` [`str`, ``any``]]: The shape specification for use \
                with GSD files for visualization.

        This is not meant to be used for access to shape information in Python.
        See the attribute ``shape`` for programatic assess. Use this property to
        log shape for visualization and storage through the GSD file type.
        """
        return self._return_type_shapes()
