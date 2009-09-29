# -*- coding: iso-8859-1 -*-
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
# Maintainer: joaander / All Developers are free to add commands for new features

## \package hoomd_script.dump
# \brief Commands that %dump particles to files
#
# Commands in the dump package write the system state out to a file every 
# \a period time steps. Check the documentation for details on which file format 
# each command writes.

import hoomd;
import globals;
import analyze;
import sys;
import util;

## Writes simulation snapshots in the HOOMD XML format
#
# Every \a period time steps, a new file will be created. The state of the 
# particles at that time step is written to the file in the HOOMD XML format.
#
# \sa \ref page_xml_file_format
class xml(analyze._analyzer):
    ## Initialize the hoomd_xml writer
    #
    # \param filename (optional) Base of the file name
    # \param period (optional) Number of time steps between file dumps
    # 
    # \b Examples:
    # \code
    # dump.xml(filename="atoms.dump", period=1000)
    # xml = dump.xml(filename="particles", period=1e5)
    # xml = dump.xml()
    # \endcode
    #
    # If period is set, a new file will be created every \a period steps. The time step at which 
    # the file is created is added to the file name in a fixed width format to allow files to easily 
    # be read in order. I.e. the write at time step 0 with \c filename="particles" produces the file 
    # \c particles.0000000000.xml
    #
    # By default, only particle positions are output to the dump files. This can be changed
    # with set_params().
    #
    # If \a period is not specified, then no periodic updates will occur. Instead, the write()
    # command must be executed to write an output file.
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename="dump", period=None):
        util.print_status_line();
    
        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.HOOMDDumpWriter(globals.system_definition, filename);
        
        if period != None:
            self.setupAnalyzer(period);
            self.enabled = False;
            self.prev_period = 1;

    ## Change xml write parameters
    #
    # \param all (if true) Enables the output of all optional parameters below
    # \param position (if set) Set to True/False to enable/disable the output of particle positions in the xml file
    # \param image (if set) Set to True/False to enable/disable the output of particle images in the xml file
    # \param velocity (if set) Set to True/False to enable/disable the output of particle velocities in the xml file
    # \param mass (if set) Set to True/False to enable/disable the output of particle masses in the xml file
    # \param diameter (if set) Set to True/False to enable/disable the output of particle diameters in the xml file
    # \param type (if set) Set to True/False to enable/disable the output of particle types in the xml file
    # \param wall (if set) Set to True/False to enable/disable the output of walls in the xml file
    # \param bond (if set) Set to True/False to enable/disable the output of bonds in the xml file
    # \param angle (if set) Set to True/False to enable/disable the output of angles in the xml file
    # \param dihedral (if set) Set to True/False to enable/disable the output of dihedrals in the xml file
    # \param improper (if set) Set to True/False to enable/disable the output of impropers in the xml file
    #
    # Using set_params() requires that the %dump was saved in a variable when it was specified.
    # \code
    # xml = dump.xml(filename="particles", period=1e5)
    # \endcode
    #
    # \b Examples:
    # \code
    # xml.set_params(type=False)
    # xml.set_params(position=False, type=False, velocity=True)
    # xml.set_params(type=True, position=True)
    # xml.set_params(position=True, wall=True)
    # xml.set_params(bond=True)
    # xml.set_params(all=True)
    # \endcode
    def set_params(self, all=None, position=None, image=None, velocity=None, mass=None, diameter=None, type=None, wall=None, bond=None, angle=None, dihedral=None, improper=None):
        util.print_status_line();
    
        # check that proper initialization has occurred
        if self.cpp_analyzer == None:
            print >> sys.stderr, "\n***Error! Bug in hoomd_script: cpp_analyzer not set, please report\n";
            raise RuntimeError('Error setting xml parameters');
            
        if all:
            position = image = velocity = mass = diameter = type = wall = bond = angle = dihedral = improper = True;

        if position != None:
            self.cpp_analyzer.setOutputPosition(position);

        if image != None:
            self.cpp_analyzer.setOutputImage(image);

        if velocity != None:
            self.cpp_analyzer.setOutputVelocity(velocity);
            
        if mass != None:
            self.cpp_analyzer.setOutputMass(mass);
            
        if diameter != None:
            self.cpp_analyzer.setOutputDiameter(diameter);
            
        if type != None:
            self.cpp_analyzer.setOutputType(type);
            
        if wall != None:
            self.cpp_analyzer.setOutputWall(wall);
            
        if bond != None:
            self.cpp_analyzer.setOutputBond(bond);

        if angle != None:
            self.cpp_analyzer.setOutputAngle(angle);
            
        if dihedral != None:
            self.cpp_analyzer.setOutputDihedral(dihedral);
            
        if improper != None:
            self.cpp_analyzer.setOutputImproper(improper);
            
    ## Write a file at the current time step
    #
    # \param filename File name to write to
    #
    # The periodic file writes can be temporarily overridden and a file with any file name
    # written at the current time step.
    #
    # Executing write() requires that the %dump was saved in a variable when it was specified.
    # \code
    # xml = dump.xml()
    # \endcode
    #
    # \b Examples:
    # \code
    # xml.write(filename="start.xml")
    # \endcode
    def write(self, filename):
        util.print_status_line();
        
        # check that proper initialization has occured
        if self.cpp_analyzer == None:
            print >> sys.stderr, "\n***Error! Bug in hoomd_script: cpp_analyzer not set, please report\n";
            raise RuntimeError('Error writing xml');
        
        self.cpp_analyzer.writeFile(filename, globals.system.getCurrentTimeStep());

## Writes a simulation snapshot in the MOL2 format
#
# Every \a period time steps, a new file will be created. The state of the 
# particles at that time step is written to the file in the MOL2 format.
#
# The intended usage is to use write() to generate a single structure file that 
# can be used by VMD for reading in particle names and %bond topology Use in 
# conjunction with dump.dcd for reading the full simulation trajectory into VMD.
class mol2(analyze._analyzer):
    ## Initialize the mol2 writer
    #
    # \param filename (optional) Base of the file name
    # \param period (optional) Number of time steps between file dumps
    # 
    # \b Examples:
    # \code
    # dump.mol2(filename="atoms.dump", period=1000)
    # mol2 = dump.mol2(filename="particles", period=1e5)
    # mol2 = dump.mol2()
    # \endcode
    #
    # If period is set, a new file will be created every \a period steps. The time step at which 
    # the file is created is added to the file name in a fixed width format to allow files to easily 
    # be read in order. I.e. the write at time step 0 with \c filename="particles" produces the file 
    # \c particles.0000000000.mol2
    #
    # If \a period is not specified, then no periodic updates will occur. Instead, the write()
    # command must be executed to write an output file.
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename="dump", period=None):
        util.print_status_line();
    
        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.MOL2DumpWriter(globals.system_definition, filename);
        
        if period != None:
            self.setupAnalyzer(period);
            self.enabled = False;
            self.prev_period = 1;
                
    ## Write a file at the current time step
    #
    # \param filename File name to write to
    #
    # The periodic file writes can be temporarily overridden and a file with any file name
    # written at the current time step.
    #
    # Executing write() requires that the %dump was saved in a variable when it was specified.
    # \code
    # mol2 = dump.mol2()
    # \endcode
    #
    # \b Examples:
    # \code
    # mol2.write(filename="start.mol2")
    # \endcode
    def write(self, filename):
        util.print_status_line();
        
        # check that proper initialization has occured
        if self.cpp_analyzer == None:
            print >> sys.stderr, "\n***Error! Bug in hoomd_script: cpp_analyzer not set, please report\n";
            raise RuntimeError('Error writing pdb');
        
        self.cpp_analyzer.writeFile(filename);

    
## Writes simulation snapshots in the DCD format
#
# Every \a period time steps a new simulation snapshot is written to the 
# specified file in the DCD file format. DCD only stores particle positions
# but is decently space efficient and extremely fast to read and write. VMD
# can load 100's of MiB of trajectory data in mere seconds.
#
# Use in conjunction with dump.mol2 so that VMD has information on the
# particle names and %bond topology.
#
# Due to constraints of the DCD file format, once you stop writing to
# a file via disable(), you cannot continue writing to the same file,
# nor can you change the period of the %dump at any time. Either of these tasks 
# can be performed by creating a new %dump file with the needed settings.
class dcd(analyze._analyzer):
    ## Initialize the dcd writer
    #
    # \param filename File name to write to
    # \param period Number of time steps between file dumps
    # \param overwrite When False, (the default) an existing DCD file will be appended to. When True, an existing DCD file \a filename will be overwritten.
    # 
    # \b Examples:
    # \code
    # dump.dcd(filename="trajectory.dcd", period=1000)<br>
    # dcd = dump.dcd(filename"data/dump.dcd", period=1000)
    # \endcode
    #
    # \warning 
    # When you use dump.dcd to append to an existing dcd file
    # - The period must be the same or the time data in the file will not be consistent.
    # - dump.dcd will not write out data at time steps that already are present in the dcd file to maintain a consistent timeline
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename, period, overwrite=False):
        util.print_status_line();
        
        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        reported_period = period;
        if type(period) != type(1):
            reported_period = 1000;
            
        self.cpp_analyzer = hoomd.DCDDumpWriter(globals.system_definition, filename, int(reported_period), overwrite);
        self.setupAnalyzer(period);
    
    def enable(self):
        util.print_status_line();
        
        if self.enabled == False:
            print >> sys.stderr, "\n***Error! you cannot re-enable DCD output after it has been disabled\n";
            raise RuntimeError('Error enabling updater');
    
    def set_period(self, period):
        util.print_status_line();
        
        print >> sys.stderr, "\n***Error! you cannot change the period of a dcd dump writer\n";
        raise RuntimeError('Error changing updater period');


## Writes simulation snapshots in the PBD format
#
# Every \a period time steps, a new file will be created. The state of the 
# particles at that time step is written to the file in the PDB format.
#
class pdb(analyze._analyzer):
    ## Initialize the pdb writer
    #
    # \param filename (optional) Base of the file name
    # \param period (optional) Number of time steps between file dumps
    #
    # \b Examples:
    # \code
    # dump.pdb(filename="atoms.dump", period=1000)
    # pdb = dump.pdb(filename="particles", period=1e5)
    # pdb = dump.pdb()
    # \endcode
    #
    # If \a period is specified, a new file will be created every \a period steps. The time step 
    # at which the file is created is added to the file name in a fixed width format to allow 
    # files to easily be read in order. I.e. the write at time step 0 with \c filename="particles" produces 
    # the file \c particles.0000000000.pdb
    #
    # By default, only particle positions are output to the dump files. This can be changed
    # with set_params().
    #
    # If \a period is not specified, then no periodic updates will occur. Instead, the write()
    # command must be executed to write an output file.
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename="dump", period=None):
        util.print_status_line();
    
        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.PDBDumpWriter(globals.system_definition, filename);
        
        if period != None:
            self.setupAnalyzer(period);
            self.enabled = False;
            self.prev_period = 1;
            
    ## Change mol2 write parameters
    #
    # \param bond (if set) Set to True/False to enable/disable the output of bonds in the mol2 file
    #
    # Using set_params() requires that the %dump was saved in a variable when it was specified.
    # \code
    # pdb = dump.pdb(filename="particles", period=1e5)
    # \endcode
    #
    # \b Examples:
    # \code
    # pdb.set_params(bond=True)
    # \endcode
    def set_params(self, bond=None):
        util.print_status_line();
    
        # check that proper initialization has occured
        if self.cpp_analyzer == None:
            print >> sys.stderr, "\n***Error! Bug in hoomd_script: cpp_analyzer not set, please report\n";
            raise RuntimeError('Error setting pdb parameters');
            
        if bond != None:
            self.cpp_analyzer.setOutputBond(bond);
    
    ## Write a file at the current time step
    #
    # \param filename File name to write to
    #
    # The periodic file writes can be temporarily overridden and a file with any file name
    # written at the current time step.
    #
    # Executing write() requires that the %dump was saved in a variable when it was specified.
    # \code
    # pdb = dump.pdb()
    # \endcode
    #
    # \b Examples:
    # \code
    # pdb.write(filename="start.pdb")
    # \endcode
    def write(self, filename):
        util.print_status_line();
        
        # check that proper initialization has occured
        if self.cpp_analyzer == None:
            print >> sys.stderr, "\n***Error! Bug in hoomd_script: cpp_analyzer not set, please report\n";
            raise RuntimeError('Error writing pdb');
        
        self.cpp_analyzer.writeFile(filename);
