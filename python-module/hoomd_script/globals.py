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

# Maintainer: joaander

## \package hoomd_script.globals
# \brief Global variables
#
# To present a simple procedural user interface, hoomd_script
# needs to track many variables globally. These are stored here.
# 
# User scripts are not intended to access these variables. However,
# there may be some special cases where it is needed. Any variable
# defined here can be accessed in a user script by prepending
# "globals." to the variable name. For example, to access the 
# global SystemDefinition, a user script can access \c globals.system_definition .

## Global variable that holds the execution configuration for reference by the python API
exec_conf = None;

## Global variable that holds the SystemDefinition shared by all parts of hoomd_script
system_definition = None;

## Global variable that holds the System shared by all parts of hoomd_script
system = None;

## Global variable that tracks the all of the force computes specified in the script so far
forces = [];

## Global variable that tracks the all of the constraint force computes specified in the script so far
constraint_forces = [];

## Global variable that tracks all the integration methods that have been specified in the script so far
integration_methods = [];

## Global variable tracking the last _integrator set
integrator = None;

## Global variable tracking the system's neighborlist
neighbor_list = None;

## Global variable tracking all the loggers that have been created
loggers = [];

## Global variable tracking all the compute thermos that have been created
thermos = [];

## Cached all group
group_all = None;

## Global options
options = None;

## Global Messenger
# \note This is initialized to a default messenger on load so that python code may have a unified path for sending
# messages
msg = hoomd.Messenger();

## \internal
# \brief Clears all global variables to default values
# \details called by hoomd_script.reset()
def clear():
    global system_definition, system, forces, constraint_forces, external_forces, integration_methods, integrator, neighbor_list, loggers, thermos;
    global group_all, exec_conf;
    
    exec_conf = None;
    system_definition = None;
    system = None;
    forces = [];
    constraint_forces = [];
    external_forces = [];
    integration_methods = [];
    integrator = None;
    neighbor_list = None;
    loggers = [];
    thermos = [];
    group_all = None;
    
    import __main__;
    __main__.sorter = None;
    __main__.nlist = None;

