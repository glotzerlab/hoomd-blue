# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define LoadBalancer."""

from hoomd.data.parameterdicts import ParameterDict
from hoomd.operation import Tuner
from hoomd import _hoomd
import hoomd


class LoadBalancer(Tuner):
    r"""Adjusts the boundaries of the domain decomposition.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the timesteps on which to
            perform load balancing.
        x (bool): Balance the **x** direction when `True`.
        y (bool): Balance the **y** direction when `True`.
        z (bool): Balance the **z** direction when `True`.
        tolerance (float): Load imbalance tolerance.
        max_iterations (int): Maximum number of iterations to
            attempt in a single step.

    `LoadBalancer` adjusts the boundaries of the MPI domains to distribute
    the particle load close to evenly between them. The load imbalance is
    defined as the number of particles owned by a rank divided by the average
    number of particles per rank if the particles had a uniform distribution:

    .. math::

        I = \frac{N_i}{N / P}

    where :math:`N_i` is the number of particles on rank :math:`i`, :math:`N` is
    the total number of particles, and :math:`P` is the number of ranks.

    In order to adjust the load imbalance, `LoadBalancer` scales by the inverse
    of the imbalance factor. To reduce oscillations and communication overhead,
    it does not move a domain more than 5% of its current size in a single
    rebalancing step, and not more than half the distance to its neighbors.

    Simulations with interfaces (so that there is a particle density gradient)
    or clustering should benefit from load balancing. The potential speedup is
    :math:`I-1.0`, so that if the largest imbalance is 1.4, then the user can
    expect a 40% speedup in the simulation. This is of course an estimate that
    assumes that all algorithms are linear in :math:`N`, all GPUs are fully
    occupied, and the simulation is limited by the speed of the slowest
    processor. If you have a simulation where, for example, some particles have
    significantly more pair force neighbors than others, this estimate of the
    load imbalance may not produce the optimal results.

    A load balancing adjustment is only performed when the maximum load
    imbalance exceeds a *tolerance*. The ideal load balance is 1.0, so setting
    *tolerance* less than 1.0 will force an adjustment every update. The load
    balancer can attempt multiple iterations of balancing on each update, and up
    to *maxiter* attempts can be made. The optimal values of update and
    *maxiter* will depend on your simulation.

    Load balancing can be performed independently and sequentially for each
    dimension of the simulation box. A small performance increase may be
    obtained by disabling load balancing along dimensions that are known to be
    homogeneous.  For example, if there is a planar vapor-liquid interface
    normal to the :math:`z` axis, then it may be advantageous to disable
    balancing along :math:`x` and :math:`y`.

    In systems that are well-behaved, there is minimal overhead of balancing
    with a small update. However, if the system is not capable of being balanced
    (for example, due to the density distribution or minimum domain size),
    having a small update and high *maxiter* may lead to a large performance
    loss. In such systems, it is currently best to either balance infrequently
    or to balance once in a short test run and then set the decomposition
    statically in a separate initialization.

    Balancing is ignored if there is no domain decomposition available (MPI is
    not built or is running on a single rank).

    Attributes:
        trigger (hoomd.trigger.Trigger): Select the timesteps on which to
            perform load balancing.
        x (bool): Balance the **x** direction when `True`.
        y (bool): Balance the **y** direction when `True`.
        z (bool): Balance the **z** direction when `True`.
        tolerance (float): Load imbalance tolerance.
        max_iterations (int): Maximum number of iterations to
            attempt in a single step.
    """

    def __init__(self,
                 trigger,
                 x=True,
                 y=True,
                 z=True,
                 tolerance=1.02,
                 max_iterations=1):
        super().__init__(trigger)

        defaults = dict(x=x,
                        y=y,
                        z=z,
                        tolerance=tolerance,
                        max_iterations=max_iterations)
        load_balancer_params = ParameterDict(x=bool,
                                             y=bool,
                                             z=bool,
                                             max_iterations=int,
                                             tolerance=float)
        self._param_dict.update(load_balancer_params)
        self._param_dict.update(defaults)

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_cls = getattr(_hoomd, 'LoadBalancerGPU')
        else:
            cpp_cls = getattr(_hoomd, 'LoadBalancer')

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                self.trigger)
