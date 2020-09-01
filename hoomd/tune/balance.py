from hoomd.operation import _Tuner
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType
from hoomd.trigger import Trigger
from hoomd import _hoomd


class LoadBalancer(_Tuner):
    R""" Adjusts the boundaries of a domain decomposition on a regular 3D grid.

    Args:
        trigger (hoomd.trigger.Trigger): A ``Trigger`` object to activate the
            `LoadBalancer`. If passed an integer, a
            :py:class:`hoomd.trigger.Periodic` trigger will be used with the
            integer as its period.
        x (:obj:`bool`, optional): If True, balance in x dimension, defaults to
            True.
        y (:obj:`bool`, optional): If True, balance in y dimension, defaults to
            True.
        z (:obj:`bool`, optional): If True, balance in z dimension, defaults to
            True.
        tolerance (:obj:`float`, optional): Load imbalance tolerance (if <= 1.0,
            balance every step), defaults to 1.02.
        max_iterations (:obj:`int`, optional): Maximum number of iterations to
            attempt in a single step, defaults to 1.

    Every ``trigger`` activation, the boundaries of the processor domains are
    adjusted to distribute the particle load close to evenly between them. The
    load imbalance is defined as the number of particles owned by a rank divided
    by the average number of particles per rank if the particles had a uniform
    distribution:

    .. math::

        I = \frac{N(i)}{N / P}

    where :math:` N(i) ` is the number of particles on processor :math:`i`,
    :math:`N` is the total number of particles, and :math:`P` is the number of
    ranks.

    In order to adjust the load imbalance, the sizes are rescaled by the inverse
    of the imbalance factor. To reduce oscillations and communication overhead,
    a domain cannot move more than 5% of its current size in a single
    rebalancing step, and the edge of a domain cannot move more than half the
    distance to its neighbors.

    Simulations with interfaces (so that there is a particle density gradient)
    or clustering should benefit from load balancing. The potential speedup is
    roughly :math:`I-1.0`, so that if the largest imbalance is 1.4, then the
    user can expect a roughly 40% speedup in the simulation. This is of course
    an estimate that assumes that all algorithms are roughly linear in
    :math:`N`, all GPUs are fully occupied, and the simulation is limited by the
    speed of the slowest processor. It also assumes that all particles roughly
    equal. If you have a simulation where, for example, some particles have
    significantly more pair force neighbors than others, this estimate of the
    load imbalance may not produce the optimal results.

    A load balancing adjustment is only performed when the maximum load
    imbalance exceeds a *tolerance*. The ideal load balance is 1.0, so setting
    *tolerance* less than 1.0 will force an adjustment every *period*. The load
    balancer can attempt multiple iterations of balancing every *period*, and up
    to *maxiter* attempts can be made. The optimal values of *period* and
    *maxiter* will depend on your simulation.

    Load balancing can be performed independently and sequentially for each
    dimension of the simulation box. A small performance increase may be
    obtained by disabling load balancing along dimensions that are known to be
    homogeneous.  For example, if there is a planar vapor-liquid interface
    normal to the :math:`z` axis, then it may be advantageous to disable
    balancing along :math:`x` and :math:`y`.

    In systems that are well-behaved, there is minimal overhead of balancing
    with a small *period*. However, if the system is not capable of being
    balanced (for example, due to the density distribution or minimum domain
    size), having a small *period* and high *maxiter* may lead to a large
    performance loss. In such systems, it is currently best to either balance
    infrequently or to balance once in a short test run and then set the
    decomposition statically in a separate initialization.

    Balancing is ignored if there is no domain decomposition available (MPI is
    not built or is running on a single rank).
    """

    def __init__(self, trigger, x=True, y=True, z=True, tolerance=1.02,
                 max_iterations=1):
        defaults = dict(x=x, y=y, z=z, tolerance=tolerance,
                        max_iterations=max_iterations, trigger=trigger)
        self._param_dict = ParameterDict(
            x=bool, y=bool, z=bool, max_iterations=int, tolerance=float,
            trigger=Trigger)
        self._param_dict.update(defaults)

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_cls = getattr(_hoomd, 'LoadBalancerGPU')
        else:
            cpp_cls = getattr(_hoomd, 'LoadBalancer')
        self._cpp_obj = cpp_cls(
            sim.state._cpp_sys_def,
            sim._cpp_sys.getCommunicator().getDomainDecomposition(),
            self.trigger)

        super()._attach()
