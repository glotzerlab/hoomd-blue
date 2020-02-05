# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

from hoomd import _hoomd


class Variant(_hoomd.Variant):
    pass


class ConstantVariant(_hoomd.VariantConstant, Variant):
    def __init__(self, value):
        _hoomd.VariantConstant.__init__(self, value)
