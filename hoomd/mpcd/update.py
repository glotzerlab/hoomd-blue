# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD updaters.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

"""

import hoomd
from hoomd.mpcd import _mpcd
from hoomd.operation import Updater
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import log
from hoomd.data.typeconverter import OnlyTypes, positive_real

import math
import inspect


class ReverseNonequilibriumShearFlow(Updater):
    r"""Reverse nonequilibrium shear flow.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the time steps on which to
            to swap momentum.

        num_swaps (int): Maximum number of pairs to swap per update.

        slab_width (float): Width of momentum-exchange slabs.

        target_momentum (float): Target magnitude of momentum for swapped
            particles (must be positive).

    `ReverseNonequilibriumShearFlow` generates a bidirectional shear flow in
    *x* by imposing a momentum flux on MPCD particles in *y*. Particles are
    selected from two momentum-exchange slabs with normal to *y*, width *w*,
    and separated by :math:`L_y/2`. The lower slab accordingly has particles
    with :math:`-L_y/2 \le y < L_y/2 + w`, while the upper slab has particles
    with :math:`0 \le y < w`.

    MPCD particles are sorted into these slabs, and the particles whose *x*
    momenta are closest to the `target_momentum` :math:`p_0` in the lower slab
    and :math:`-p_0` in the upper slab are selected for a momentum swap.
    Up to `num_swaps` swaps are executed each time.

    The amount of momentum transferred from the lower slab to the upper slab
    is accumulated into `summed_exchanged_momentum`. This quantity can be used
    to calculate the momentum flux and, in conjunction with the shear velocity
    field that is generated, determine the shear viscosity.

    .. rubric:: Examples:

    To implement the method as originally proposed by `Müller-Plathe`_,
    only the fastest particle and the slowest particle in the momentum-exchange
    slabs are swapped. Set `num_swaps` to 1 and `target_momentum` at its
    default value of infinity.

    .. code-block:: python

            flow = hoomd.mpcd.update.ReverseNonequilibriumShearFlow(
                trigger=1, num_swaps=1, slab_width=1
            )
            simulation.operations.updaters.append(flow)

    An alternative approach proposed by `Tenney and Maginn`_ swaps particles
    that are instead closest to the `target_momentum`, typically requiring more
    swaps per update.

    .. code-block:: python

            flow = hoomd.mpcd.update.ReverseNonequilibriumShearFlow(
                trigger=1, num_swaps=10, slab_width=1, target_momentum=5
            )
            simulation.operations.updaters.append(flow)

    {inherited}

    ----------

    **Members defined in** `ReverseNonequilibriumShearFlow`:

    Attributes:
        num_swaps (int): Maximum number of times to swap momentum per update.

            .. rubric:: Example:

            .. code-block:: python

                flow.num_swaps = 10

        slab_width (float): Width of momentum-exchange slabs.

            .. rubric:: Example:

            .. code-block:: python

                flow.slab_width = 1

        target_momentum (float): Target momentum for swapped particles.

            .. rubric:: Example:

            .. code-block:: python

                flow.target_momentum = 5

    .. _Müller-Plathe: https://doi.org/10.1103/PhysRevE.59.4894
    .. _Tenney and Maginn: https://doi.org/10.1063/1.3276454

    """

    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(Updater._doc_inherited)
    )

    def __init__(self, trigger, num_swaps, slab_width, target_momentum=math.inf):
        super().__init__(trigger)

        param_dict = ParameterDict(
            num_swaps=int(num_swaps),
            slab_width=float(slab_width),
            target_momentum=OnlyTypes(float, preprocess=positive_real),
        )
        param_dict["target_momentum"] = target_momentum
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            class_ = _mpcd.ReverseNonequilibriumShearFlowGPU
        else:
            class_ = _mpcd.ReverseNonequilibriumShearFlow

        self._cpp_obj = class_(
            self._simulation.state._cpp_sys_def,
            self.trigger,
            self.num_swaps,
            self.slab_width,
            self.target_momentum,
        )

        super()._attach_hook()

    @log(category="scalar", requires_run=True)
    def summed_exchanged_momentum(self):
        r"""float: Total momentum exchanged.

        This quantity is the total momentum exchanged from the lower slab
        to the upper slab.

        """
        return self._cpp_obj.summed_exchanged_momentum


__all__ = [
    "ReverseNonequilibriumShearFlow",
]
