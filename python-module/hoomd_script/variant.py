# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
# Iowa State University and The Regents of the University of Michigan All rights
# reserved.

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

# * All publications based on HOOMD-blue, including any reports or published
# results obtained, in whole or in part, with HOOMD-blue, will acknowledge its use
# according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website
# at: http://codeblue.umich.edu/hoomd-blue/.

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

import hoomd;
import globals;
import sys;
import init;

## \package hoomd_script.variant
# \brief Commands for specifying values that vary over time
#
# This package contains various commands for creating quantities that can vary
# smoothly over the course of a simulation. For example, the temperature in
# update.nvt can be set to a variant.linear_interp in order to slowly heat
# or cool the system over a long simulation.

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
        if not init.is_initialized():
            print >> sys.stderr, "\n***Error! Cannot create a variant before initialization\n";
            raise RuntimeError('Error creating variant');
        
        self.cpp_variant = None;
        
## \internal
# \brief A constant "variant"
#
# This is just a placeholder for a constant value. It does not need to be documented
# as all hoomd_script commands that take in variants should use _setup_variant_input()
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
        
        # create the c++ mirror class
        self.cpp_variant = hoomd.VariantConst(val);
        self.cpp_variant.setOffset(globals.system.getCurrentTimeStep());
        
## Linearly interpolated variant
#
# variant.linear_interp creates a time-varying quantity where the value at each time step
# is determined by linear interpolation between a given set of points. 

# At time steps before the
# initial point, the value is identical to the value at the first given point. At time steps 
# after the final point, the value is identical to the value at the last given point. All points
# between are determined by linear interpolation.
#
# By default, a time step for a given point is referenced to the current time step of the simulation.
# For example,
# \code
# init.create_random(N=1000, phi_p=0.2)
# ...
# run(1000)
# variant.linear_interp(...)
# run(1000)
# \endcode
# A value specified at time 0 in the shown linear_interp is set at the actual \b absolute time step
# 1000. To say it another way, time for validate.linear_interp starts counting from 0 right
# at the time of creation. The point where 0 is defined can be changed by setting the \z zero parameter in the 
# command that specifies the linear_interp.
#
# See __init__() for the syntax which the set values can be specified.
class linear_interp(_variant):
    ## Specify a linearly interpolated %variant
    #
    # \param points Set points in the linear interpolation (see below)
    # \param zero Specify absolute time step number location for 0 in \a points. Use 'now' to indicate the current step.
    #
    # \a points is a list of (time, set value) tuples. For example, to specify
    # a series of points that goes from 10 at time step 0 to 20 at time step 100 and then
    # back down to 5 at time step 200:
    # \code 
    # points = [(0, 10), (100, 20), (200, 5)]
    # \endcode
    # Any number of points can be specified in any order. However, listing them 
    # monotonically increasing in time will result in a much more human readable set 
    # of values.
    #
    # \b Examples:
    # \code
    # L = variant.linear_interp(points = [(0, 10), (100, 20), (200, 5)])
    # V = variant.linear_interp(points = [(0, 10), (1e6, 20)], zero=80000)
    # integrate.nvt(group=all, tau = 0.5, 
    #     T = variant.linear_interp(points = [(0, 1.0), (1e5, 2.0)])
    # \endcode
    def __init__(self, points, zero='now'):
        # initialize the base class
        _variant.__init__(self);
        
        # create the c++ mirror class
        self.cpp_variant = hoomd.VariantLinear();
        if zero == 'now':
            self.cpp_variant.setOffset(globals.system.getCurrentTimeStep());
        else:
            # validate zero
            if zero < 0:
                print >> sys.stderr, "\n***Error! Cannot create a linear_interp variant with a negative zero\n";
                raise RuntimeError('Error creating variant');
            if zero > globals.system.getCurrentTimeStep():
                print >> sys.stderr, "\n***Error! Cannot create a linear_interp variant with a zero in the future\n";
                raise RuntimeError('Error creating variant');
                                
            self.cpp_variant.setOffset(zero);
        
        # set the points
        if len(points) == 0:
            print >> sys.stderr, "\n***Error! Cannot create a linear_interp variant with 0 points\n";
            raise RuntimeError('Error creating variant');
            
        for (t, v) in points:
            if t < 0:
                print >> sys.stderr, "\n***Error! Negative times are not allowed in variant.linear_interp\n";
                raise RuntimeError('Error creating variant');
        
            self.cpp_variant.setPoint(int(t), v);


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
            print >> sys.stderr, "\n***Error! Value must either be a scalar value or a the result of a variant command\n";
            raise RuntimeError('Error creating variant');
    

