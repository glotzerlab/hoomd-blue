# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

from hoomd.meta import _Operation


class _BaseIntegrator(_Operation):
    def attach(self, simulation):
        simulation._cpp_sys.setIntegrator(self._cpp_obj)
        super().attach(simulation)
