# This file exists to allow the hoomd module to import from the source checkout dir
# for use when building the sphinx documentation.

import itertools

_exports = [];

for (potential, dim, suffix) in itertools.product(['WCA', 'SWCA'], ['2', '3'], ['', 'GPU']):
    _exports.append(potential + 'DEM' + dim + 'D' + suffix);

_exports.extend(['WCAPotential', 'SWCAPotential', 'NoFriction']);

for name in _exports:
    globals()[name] = object;
