# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.dem import _dem


class WCA(_dem.WCAPotential):
    """Parameter wrapper for a WCA-like potential."""

    def __init__(self, radius):
        super(WCA, self).__init__(radius, NoFriction())


class SWCA(_dem.SWCAPotential):
    """Parameter wrapper for a shifted WCA-like potential."""

    def __init__(self, radius):
        super(SWCA, self).__init__(radius, NoFriction())


class NoFriction(_dem.NoFriction):
    """"Parameter" wrapper for force fields with no friction
    (i.e. real physical potentials)"""

    def __init__(self):
        super(NoFriction, self).__init__()
