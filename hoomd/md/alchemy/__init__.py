# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Alchemical molecular dynamics.

Description of alchmical molecular dynamics.

Important:
    `van Anders et al. 2015 <https://doi.org/10.1021/acsnano.5b04181>`_
    describes the digital alchemy method, and `Zhou et al. 2019
    <https://doi.org/10.1080/00268976.2019.1680886>`_ describes the application
    of digital alchemy to molecular dynamics simulations. Cite both works if you
    use `hoomd.md.alchemy` in your research.

"""

from . import methods
from . import pair
