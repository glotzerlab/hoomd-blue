# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Anisotropic pair forces.

Anisotropic pair force classes apply a force, torque, and virial on every
particle in the simulation state commensurate with the potential energy:

.. math::

    U_\mathrm{pair,total} = \frac{1}{2} \sum_{i=0}^\mathrm{N_particles-1}
                      \sum_{j \ne i, (i,j) \notin \mathrm{exclusions}}
                      U_\mathrm{pair}(r_{ij}, \mathbf{q}_i, \mathbf{q}_j)

`AnisotropicPair` applies cuttoffs, exclusions, and assigns per particle
energies and virials in the same manner as `hoomd.md.pair.Pair`

`AnisotropicPair` does not support the ``'xplor'`` shifting mode or the ``r_on``
parameter.
"""

from collections.abc import Sequence
import json
from numbers import Number

from hoomd.md.pair.pair import Pair
from hoomd.logging import log
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, to_type_converter


class AnisotropicPair(Pair):
    r"""Base class anisotropic pair force.

    `AnisotropicPair` is the base class for all anisotropic pair forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

    Args:
        nlist (hoomd.md.nlist.NeighborList) : The neighbor list.
        default_r_cut (`float`, optional) : The default cutoff for the
            potential, defaults to ``None`` which means no cutoff
            :math:`[\mathrm{length}]`.
        mode (`str`, optional) : the energy shifting mode, defaults to "none".
    """

    _accepted_modes = ("none", "shift")

    def __init__(self, nlist, default_r_cut=None, mode="none"):
        super().__init__(nlist, default_r_cut, 0.0, mode)

    def _return_type_shapes(self):
        type_shapes = self._cpp_obj.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret


class Dipole(AnisotropicPair):
    r"""Screened dipole-dipole pair forces.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `Dipole` computes the (screened) interaction between pairs of
    particles with dipoles and electrostatic charges:

    .. math::

        U &= U_{dd} + U_{de} + U_{ee}

        U_{dd} &= A e^{-\kappa r}
            \left(\frac{\vec{\mu_i}\cdot\vec{\mu_j}}{r^3}
                  - 3\frac{(\vec{\mu_i}\cdot \vec{r_{ji}})
                           (\vec{\mu_j}\cdot \vec{r_{ji}})}
                          {r^5}
            \right)

        U_{de} &= A e^{-\kappa r}
            \left(\frac{(\vec{\mu_j}\cdot \vec{r_{ji}})q_i}{r^3}
                - \frac{(\vec{\mu_i}\cdot \vec{r_{ji}})q_j}{r^3}
            \right)

        U_{ee} &= A e^{-\kappa r} \frac{q_i q_j}{r}

    Note:
       All units are documented electronic dipole moments. However, `Dipole`
       can also be used to represent magnetic dipoles.

    Example::

        nl = nlist.Cell()
        dipole = md.pair.ansio.Dipole(nl, default_r_cut=3.0)
        dipole.params[('A', 'B')] = dict(A=1.0, kappa=4.0)
        dipole.mu['A'] = (4.0, 1.0, 0.0)

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
    """
    _cpp_class_name = "AnisoPotentialPairDipole"

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut, 'none')
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(A=float, kappa=float, len_keys=2))
        mu = TypeParameter('mu', 'particle_types',
                           TypeParameterDict((float, float, float), len_keys=1))
        self._extend_typeparam((params, mu))


class GayBerne(AnisotropicPair):
    r"""Gay-Berne anisotropic pair force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.
        mode (str): energy shifting/smoothing mode.

    `GayBerne` computes the Gay-Berne anisotropic pair force on every particle
    in the simulation state. This version of the Gay-Berne force supports
    identical pairs of uniaxial ellipsoids, with orientation-independent
    energy-well depth. The potential comes from the following paper `Allen et.
    al. 2006`_.

    .. _Allen et. al. 2006: https://dx.doi.org/10.1080/00268970601075238

    .. math::
        U(\vec r, \vec e_i, \vec e_j) =
        \begin{cases}
        4 \varepsilon \left[ \zeta^{-12} - \zeta^{-6} \right]
        & \zeta < \zeta_{\mathrm{cut}} \\
        0 & \zeta \ge \zeta_{\mathrm{cut}} \\
        \end{cases}

    where

    .. math::

        \zeta &= \left(\frac{r-\sigma+\sigma_{\mathrm{min}}}
                           {\sigma_{\mathrm{min}}}\right),

        \sigma^{-2} &= \frac{1}{2} \hat{\vec{r}}
            \cdot \vec{H^{-1}} \cdot \hat{\vec{r}},

        \vec{H} &= 2 \ell_\perp^2 \vec{1}
            + (\ell_\parallel^2 - \ell_\perp^2)
              (\vec{e_i} \otimes \vec{e_i} + \vec{e_j} \otimes \vec{e_j}),

    and :math:`\sigma_{\mathrm{min}} = 2 \min(\ell_\perp, \ell_\parallel)`.

    The cut-off parameter :math:`r_{\mathrm{cut}}` is defined for two particles
    oriented parallel along the **long** axis, i.e.
    :math:`\zeta_{\mathrm{cut}} = \left(\frac{r-\sigma_{\mathrm{max}}
    + \sigma_{\mathrm{min}}}{\sigma_{\mathrm{min}}}\right)`
    where :math:`\sigma_{\mathrm{max}} = 2 \max(\ell_\perp, \ell_\parallel)` .

    The quantities :math:`\ell_\parallel` and :math:`\ell_\perp` denote the
    semi-axis lengths parallel and perpendicular to particle orientation.

    Example::

        nl = nlist.Cell()
        gay_berne = md.pair.aniso.GayBerne(nlist=nl, default_r_cut=2.5)
        gay_berne.params[('A', 'A')] = dict(epsilon=1.0, lperp=0.45, lpar=0.5)
        gay_berne.r_cut[('A', 'B')] = 2 ** (1.0 / 6.0)

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
    r"""Anistropic LJ force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[length]`.

    `ALJ` computes the Lennard-Jones force between anisotropic particles as
    described in `Ramasubramani, V.  et al. 2020`_, using the formula:

    .. _Ramasubramani, V.  et al. 2020: https://doi.org/10.1063/5.0019735

    .. math::

        U(r, r_c) = U_0(r) + U_c(r_c)

    The first term is the central interaction :math:`U_0`, the standard
    center-center interaction between two Lennard-Jones particles with
    center-center distance :math:`r`. The second term is the contact interaction
    :math:`U_c`, computed from the smallest distance between the surfaces of the
    two shapes :math:`r_c`. The central and contact interactions are defined as
    follows:

    .. math::

        &U_0(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
        \left( \frac{\sigma}{r} \right)^{6} \right]

        &U_c(r_c) = 4 \varepsilon_c(\varepsilon) \left[ \left(
        \frac{\sigma_c}{r_c} \right)^{12} - \left( \frac{\sigma_c}{r_c}
        \right)^{6} \right]

    where :math:`\varepsilon` (`epsilon <params>`) affects strength of both the
    central and contact interactions, :math:`\varepsilon_c` is an energy
    coefficient set proportional to :math:`\varepsilon` to preserve the shape,
    :math:`\sigma` is the interaction distance of the central term computed as
    the average of :math:`\sigma_i` (`sigma_i <params>`) and :math:`\sigma_j`
    (`sigma_j <params>`). Lastly, `ALJ` uses the contact ratios :math:`\beta_i`
    (`contact_ratio_i <params>`) and :math:`\beta_j` (`contact_ratio_j
    <params>`) to compute the contact sigma :math:`\sigma_c` as follows:

    .. math::

        \sigma_c &= \frac{1}{2} \left[\sigma_{ci} + \sigma_{cj} \right]

        \sigma_{ci} &= \beta_i \cdot \sigma_i

        \sigma_{cj} &= \beta_j \cdot \sigma_j

    The total potential energy is therefore the sum of two interactions, a
    central Lennard-Jones potential and a radially-shifted Lennard-Jones
    potential where the shift is anisotropic and depends on the extent of the
    shape in each direction.

    Each term has an independent cutoff at which the energy is set to zero.
    The behavior of these cutoffs is dependent on whether a user requires LJ
    or Weeks-Chandler-Anderson (WCA)-like (repulsive-only) behavior. This
    behavior is controlled using the `alpha <params>` parameter, which can
    take on the following values:

    .. list-table:: Set alpha based on range of the center-center and
       contact-contact interactions.
       :header-rows: 1
       :stub-columns: 1

       * -
         - center-center repulsive only
         - center-center full-range
       * - contact-contact repulsive only
         - alpha = 0
         - alpha = 1
       * - contact-contact full-range
         - alpha = 2
         - alpha = 3

    For polytopes, computing interactions using a single contact point leads to
    significant instabilities in the torques because the contact point can jump
    from one end of a face to another in an arbitrarily small time interval. To
    ameliorate this, the ALJ potential performs a local averaging over all the
    features associated with the closest simplices on two polytopes. This
    averaging can be turned off by setting the ``average_simplices`` key for the
    type pair to ``False``.

    Specifying only ``rounding_radii`` creates an ellipsoid, while specifying
    only ``vertices`` creates a convex polytope (set ``vertices`` and ``faces``
    to empty lists to create the ellipsoid).

    Important:
        The repulsive part of the contact interaction :math:`U_c(r_c)` prevents
        two `ALJ` particles from approaching closely, effectively rounding the
        shape by a radius :math:`\sigma_c`. For this reason, the shape written
        by `type_shapes` includes the rounding due to `rounding_radii <shape>`
        and that due to :math:`\sigma_c`.

    .. rubric:: Choosing `r_cut <hoomd.md.pair.Pair.r_cut>`:

    Set `r_cut <hoomd.md.pair.Pair.r_cut>` for each pair of particle types so
    that `ALJ` can compute interactions for all possible relative placements
    and orientations of the particles. The farthest apart two particles can be
    while still interacting depends on the value of ``alpha``.

    In the following list, the first argument to the :math:`\max` function is
    for the center-center interaction. The second argument is for the
    contact-contact interaction, where :math:`R_i` is the radius of the
    shape's minimal origin-centered bounding sphere of the particle with type
    :math:`i`.

    Let :math:`\lambda_{min} = 2^{1/6}` be the position of the potential energy
    minimum of the Lennard-Jones potential and
    :math:`\lambda_{cut}^{attractive}` be a larger value, such as 2.5 (typically
    used in isotropic LJ systems).

    * For alpha=0:

      .. math::

        r_{\mathrm{cut},ij} = \max \bigg( & \frac{\lambda_{min}}{2}
        (\sigma_i + \sigma_j), \\
        & R_i + R_j + R_{\mathrm{rounding},i} +
        R_{\mathrm{rounding},j} + \frac{\lambda_{min}}{2}
        (\beta_i \cdot \sigma_i + \beta_j \cdot \sigma_j) \bigg)

    * For alpha=1:

      .. math::

            r_{\mathrm{cut},ij} =
            \max \bigg( & \frac{\lambda_{cut}^{attractive}}{2}
            (\sigma_i + \sigma_j),  \\
            & R_i + R_j  + R_{\mathrm{rounding},i} +
            R_{\mathrm{rounding},j}+ \frac{\lambda_{min}}{2}
            (\beta_i \cdot \sigma_i + \beta_j \cdot \sigma_j) \bigg)

    * For alpha=2:

      .. math::

            r_{\mathrm{cut},ij} = \max \bigg( & \frac{\lambda_{min}}{2}
            (\sigma_i + \sigma_j)),  \\
            & R_i + R_j + R_{\mathrm{rounding},i} +
            R_{\mathrm{rounding},j} + \frac{\lambda_{cut}^{attractive}}{2}
            (\beta_i \cdot \sigma_i + \beta_j \cdot \sigma_j) \bigg)

    * For alpha=3:

      .. math::

            r_{\mathrm{cut},ij} =
            \max \bigg( & \frac{\lambda_{cut}^{attractive}}{2}
            (\sigma_i + \sigma_j),  \\
            & R_i + R_j + R_{\mathrm{rounding},i} +
            R_{\mathrm{rounding},j} + \frac{\lambda_{cut}^{attractive}}{2}
            (\beta_i \cdot \sigma_i + \beta_j \cdot \sigma_j) \bigg)

    Warning:
        Changing dimension in a simulation will invalidate this force and will
        lead to error or unrealistic behavior.

    .. py:attribute:: params

        The ALJ potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) - base energy scale
          :math:`\varepsilon` :math:`[energy]`.
        * ``sigma_i`` (`float`, **required**) - the insphere diameter of the
          first particle type, :math:`\sigma_i` :math:`[length]`.
        * ``sigma_j`` (`float`, **required**) - the insphere diameter of the
          second particle type, :math:`\sigma_j` :math:`[length]`.
        * ``alpha`` (`int`, **required**) - Integer 0-3 indicating whether or
          not to include the attractive component of the interaction (see
          above for details).
        * ``contact_ratio_i`` (`float`, **optional**) - :math:`\beta_i`, the
          ratio of the contact sphere diameter of the first type with
          ``sigma_i``.
          Defaults to 0.15.
        * ``contact_ratio_j`` (`float`, **optional**) - :math:`\beta_j`, the
          ratio of the contact sphere diameter of the second type with
          ``sigma_j``.
          Defaults to 0.15.
        * ``average_simplices`` (`bool`, **optional**) - Whether to average over
          simplices. Defaults to ``True``. See class documentation for more
          information.

        Type: `hoomd.data.typeparam.TypeParameter` [`tuple` [``particle_types``,
        ``particle_types``], `dict`]

    Note:
        While the evaluation of the potential is symmetric with respect to
        the potential parameter labels ``i`` and ``j``, the parameters which
        physically represent a specific particle type must appear in all sets
        of pair parameters which include that particle type.

    .. py:attribute:: shape

        The shape of a given type. The dictionary has the following keys per
        type:

        * ``vertices`` (`list` [`tuple` [`float`, `float`, `float`]],
          **required**) - The vertices of a convex polytope in 2 or 3
          dimensions. The third dimension in 2D is ignored.
        * ``rounding_radii`` (`tuple` [`float`, `float`, `float`] or `float`)
          - The semimajor axes of a rounding ellipsoid
          :math:`R_{\mathrm{rounding},i}`. If a single value is specified, the
          rounding ellipsoid is a sphere. Defaults to (0.0, 0.0, 0.0).
        * ``faces`` (`list` [`list` [`int`]], **required**) - The faces of the
          polyhedron specified as a list of list of integers.  The indices
          corresponding to the vertices must be ordered counterclockwise with
          respect to the face normal vector pointing outward from the origin.

        Type: `hoomd.data.typeparam.TypeParameter` [``particle_types``, `dict`]

    Example::

        nl = hoomd.md.nlist.Cell(buffer=0.4)
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
                              faces=cube_faces)

    The following example shows how to easily get the faces, with vertex indices
    properly ordered, for a shape with known vertices by using the
    `coxeter <https://coxeter.readthedocs.io/>`_ package:

    Example::

        import coxeter

        nl = hoomd.md.nlist.Cell(buffer=0.4)
        alj = hoomd.md.pair.aniso.ALJ(nl, r_cut=2.5)

        cube_verts = [[-0.5, -0.5, -0.5],
                      [-0.5, -0.5, 0.5],
                      [-0.5, 0.5, -0.5],
                      [-0.5, 0.5, 0.5],
                      [0.5, -0.5, -0.5],
                      [0.5, -0.5, 0.5],
                      [0.5, 0.5, -0.5],
                      [0.5, 0.5, 0.5]]

        cube = coxeter.shapes.ConvexPolyhedron(cube_verts)

        alj.params[("A", "A")] = dict(epsilon=2.0,
                                      sigma_i=1.0,
                                      sigma_j=1.0,
                                      alpha=1,
                                      )
        alj.shape["A"] = dict(vertices=cube.vertices,
                              faces=cube.faces)

    """

    # We don't define a _cpp_class_name since the dimension is a template
    # parameter in C++, so use an instance level attribute instead that is
    # created in _attach based on the dimension of the associated simulation.

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut, 'none')
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
                              len_keys=1,
                              _defaults={'rounding_radii': (0.0, 0.0, 0.0)}))

        self._extend_typeparam((params, shape))

    def _attach_hook(self):
        self._cpp_class_name = "AnisoPotentialPairALJ{}".format(
            "2D" if self._simulation.state.box.is2D else "3D")

        super()._attach_hook()

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
