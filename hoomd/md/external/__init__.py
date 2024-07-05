# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""External forces for molecular dynamics.

External force classes apply forces to particles that result from an external
field as a function of particle position and orientation:

.. math::

    U_\mathrm{external} = \sum_i U(\vec{r}_i, \mathbf{q}_i)
"""

from . import field
from . import wall
