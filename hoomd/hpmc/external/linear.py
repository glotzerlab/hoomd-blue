# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Linear external potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere
"""

import hoomd

from .external import External


@hoomd.logging.modify_namespace(('hpmc', 'external', 'Linear'))
class Linear(External):
    """Linear external potential (HPMC).

    Args:
        default_alpha (float): Default value for alpha
            :math:`[\\mathrm{energy}] [\\mathrm{length}]^{-1}`
        plane_origin ([`float`, `float`, `float`]): A point on the plane of 0 energy.
        plane_normal ([`float`, `float`, `float`]): A vector perpendicular to the plane of 0 energy.

    `Linear` computes a linear external potential on all particles in the simulation state:

    .. math::

        U_{\\mathrm{external},i} = \\alpha_i \\cdot \\vec{n} \\cdot
            ( \\vec{r}_i - \\vec{p} )

    where :math:`\\alpha_i` (`alpha`) is the linear energy
    coefficient , :math:`\\vec{n}` is the normal vector to the plane
    (`plane_normal`), and :math:`\\vec{p}` is the plane origin (`plane_origin`):


    .. rubric:: Example

    .. code-block:: python

        linear = hoomd.hpmc.external.Linear()
        linear.alpha['A'] = 0.2
        simulation.operations.integrator.external_potentials = [linear]

    .. py:attribute:: alpha

        The linear energy coefficient :math:`\\alpha` by particle type.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `float`]

    .. py:attribute:: plane_origin

        A point on the plane where the energy is zero.

        Type: (`float`, `float`, `float`)

    .. py:attribute:: plane_normal

        A unit length vector perpendicular to the plane where the energy is zero.

        Type: (`float`, `float`, `float`)
    """
    _cpp_class_name = "ExternalPotentialLinear"

    def __init__(self,
                 default_alpha=None,
                 plane_origin=(0, 0, 0),
                 plane_normal=(0, 1, 0)):
        if default_alpha is not None:
            default_alpha = float(default_alpha)

        alpha = hoomd.data.typeparam.TypeParameter(
            'alpha', 'particle_types',
            hoomd.data.parameterdicts.TypeParameterDict(float, len_keys=1))

        if default_alpha is not None:
            alpha.default = default_alpha

        self._add_typeparam(alpha)

        self._param_dict.update(
            hoomd.data.parameterdicts.ParameterDict(plane_origin=(float, float,
                                                                  float),
                                                    plane_normal=(float, float,
                                                                  float)))
        self.plane_origin = plane_origin
        self.plane_normal = plane_normal
