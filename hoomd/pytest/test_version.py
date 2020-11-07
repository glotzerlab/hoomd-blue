# Copyright (c) 2009-2020 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

import hoomd

def test_version():
    """Test version information."""
    assert hoomd.__version__
