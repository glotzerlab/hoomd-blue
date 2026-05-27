# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import numpy

def test_table_default_arrays():
    """Ensure setting default for a table pair potential's array param still allows setting it for specific types."""
    table = hoomd.md.pair.Table(hoomd.md.nlist.Tree(2))

    table.params.default = {'U': [1, 1, 1, 1]}
    table.params[("A", "B")] = {'U': [1, 2, 3, 4]}
    
    numpy.testing.assert_array_equal(
        table.params.default["U"],
        numpy.array([1.0, 1.0, 1.0, 1.0])
    )
    
    numpy.testing.assert_array_equal(
        table.params[("A", "B")]["U"],
        numpy.array([1.0, 2.0, 3.0, 4.0])
    )
