# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$

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
# global ParticleData, a user script can access \c globals.particle_data .

## Global variable that holds the ParticleData shared by all parts of hoomd_script
particle_data = None;

## Global variable that holds the System shared by all parts of hoomd_script
system = None;

## Global variable that tracks the all of the force computes specified in the script so far
forces = [];

## Global variable tracking the last _integrator set
integrator = None;

## Global variable tracking the system's neighborlist
neighbor_list = None;

# Storing the initializer is for setting up bonds and neighborlist exclusions. 
# It is a TEMPORARY HACK to get things running quickly. When Bonds are reorganized
# in #42, this hack will go away. Only initializers that set bonds should assign 
# themselves to this variable
## (Temporary hack) Global variable tracking the initializer used
initializer = None;

## temporary hack to get bond information to the mol2 dump
bond_compute = None;
