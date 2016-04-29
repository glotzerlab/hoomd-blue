# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
# the University of Michigan All rights reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Specify values that vary over time.

This package contains various commands for creating quantities that can vary
smoothly over the course of a simulation. For example, set the temperature in
a NVT simulation to slowly heat or cool the system over a long simulation.
"""

from hoomd import _hoomd;
import hoomd;
import sys;

## \internal
# \brief Base class for variant type
#
# _variant should not be used directly in code, it only serves as a base class
# for the other variant types.
class _variant:
    ## Does common initialization for all variants
    #
    def __init__(self):
        # check if initialization has occurred
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create a variant before initialization\n");
            raise RuntimeError('Error creating variant');

        self.cpp_variant = None;

## \internal
# \brief A constant "variant"
#
# This is just a placeholder for a constant value. It does not need to be documented
# as all hoomd commands that take in variants should use _setup_variant_input()
# which will allow a simple constant number to be passed in and automatically converted
# to variant.constant for use in setting up whatever code uses the variant.
class _constant(_variant):
    ## Specify a %constant %variant
    #
    # \param val Value of the variant
    #
    def __init__(self, val):
        # initialize the base class
        _variant.__init__(self);

        self.val = val

        # create the c++ mirror class
        self.cpp_variant = _hoomd.VariantConst(val);
        self.cpp_variant.setOffset(hoomd.context.current.system.getCurrentTimeStep());

    ## \internal
    # \brief return metadata
    def get_metadata(self):
        return self.val

class linear_interp(_variant):
    R""" Linearly interpolated variant.

    Args:
        points (list): Set points in the linear interpolation (see below)
        zero (int): Specify absolute time step number location for 0 in *points*. Use 'now' to indicate the current step.


    :py:class:`hoomd.variant.linear_interp` creates a time-varying quantity where the
    value at each time step is determined by linear interpolation between a given set of points.

    At time steps before the initial point, the value is identical to the value at the first
    given point. At time steps after the final point, the value is identical to the value at
    the last given point. All points between are determined by linear interpolation.

    Time steps given to :py:class:`hoomd.variant.linear_interp` are relative to the current
    step of the simulation, and starts counting from 0 at the time of creation. Set
    *zero* to control the relative starting point.

    *points* is a list of ``(time step, set value)`` tuples. For example, to specify
    a series of points that goes from 10 at time step 0 to 20 at time step 100 and then
    back down to 5 at time step 200::

        points = [(0, 10), (100, 20), (200, 5)]

    Any number of points can be specified in any order. However, listing them
    monotonically increasing in time will result in a much more human readable set
    of values.

    Examples::

        L = variant.linear_interp(points = [(0, 10), (100, 20), (200, 5)])
        V = variant.linear_interp(points = [(0, 10), (1e6, 20)], zero=80000)
        integrate.nvt(group=all, tau = 0.5,
            T = variant.linear_interp(points = [(0, 1.0), (1e5, 2.0)])
    """
    def __init__(self, points, zero='now'):
        # initialize the base class
        _variant.__init__(self);

        # create the c++ mirror class
        self.cpp_variant = _hoomd.VariantLinear();
        if zero == 'now':
            self.cpp_variant.setOffset(hoomd.context.current.system.getCurrentTimeStep());
        else:
            # validate zero
            if zero < 0:
                hoomd.context.msg.error("Cannot create a linear_interp variant with a negative zero\n");
                raise RuntimeError('Error creating variant');
            if zero > hoomd.context.current.system.getCurrentTimeStep():
                hoomd.context.msg.error("Cannot create a linear_interp variant with a zero in the future\n");
                raise RuntimeError('Error creating variant');

            zero = int(zero)
            self.cpp_variant.setOffset(zero);

        # set the points
        if len(points) == 0:
            hoomd.context.msg.error("Cannot create a linear_interp variant with 0 points\n");
            raise RuntimeError('Error creating variant');

        for (t, v) in points:
            if t < 0:
                hoomd.context.msg.error("Negative times are not allowed in variant.linear_interp\n");
                raise RuntimeError('Error creating variant');

            self.cpp_variant.setPoint(int(t), v);

        # store metadata
        self.points = points

    ## \internal
    # \brief return metadata
    def get_metadata(self):
        return self.points

## \internal
# \brief Internal helper function to aid in setting up variants
#
# For backwards compatibility and convenience, anything that takes in a Variant should
# also automatically take in a constant number. This method will take the values passed
# in by the user and turn it into a variant._constant if it is a number. Otherwise,
# it will return the variant unchanged.
def _setup_variant_input(v):
    if isinstance(v, _variant):
        return v;
    else:
        try:
            return _constant(float(v));
        except ValueError:
            hoomd.context.msg.error("Value must either be a scalar value or a the result of a variant command\n");
            raise RuntimeError('Error creating variant');
