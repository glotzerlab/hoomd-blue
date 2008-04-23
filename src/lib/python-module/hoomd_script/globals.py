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
# defined here can be accessed in a user script by preprending
# "globals." to the variable name. For example, to access the 
# global ParticleData, a user script can access \c globals.particle_data .

## Global variable that holds the ParticleData shared by all parts of hoomd_script
particle_data = None;

## Global variable that holds the System shared by all parts of hoomd_script
system = None;


