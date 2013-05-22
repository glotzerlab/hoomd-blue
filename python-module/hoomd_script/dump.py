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

# Maintainer: joaander / All Developers are free to add commands for new features

## \package hoomd_script.dump
# \brief Commands that %dump particles to files
#
# Commands in the dump package write the system state out to a file every 
# \a period time steps. Check the documentation for details on which file format 
# each command writes.

import hoomd;
from hoomd_script import globals;
from hoomd_script import analyze;
import sys;
from hoomd_script import util;
from hoomd_script import group as hs_group;

## Writes simulation snapshots in the HOOMD XML format
#
# Every \a period time steps, a new file will be created. The state of the 
# particles at that time step is written to the file in the HOOMD XML format.
# All values are written in native HOOMD-blue units, see \ref page_units for more information.
#
# \sa \ref page_xml_file_format
# \MPI_SUPPORTED
class xml(analyze._analyzer):
    ## Initialize the hoomd_xml writer
    #
    # \param filename (optional) Base of the file name
    # \param period (optional) Number of time steps between file dumps
    # \param params (optional) Any number of parameters that set_params() accepts
    # 
    # \b Examples:
    # \code
    # dump.xml(filename="atoms.dump", period=1000)
    # xml = dump.xml(filename="particles", period=1e5)
    # xml = dump.xml(filename="test.xml", vis=True)
    # \endcode
    #
    # If period is set, a new file will be created every \a period steps. The time step at which 
    # the file is created is added to the file name in a fixed width format to allow files to easily 
    # be read in order. I.e. the write at time step 0 with \c filename="particles" produces the file 
    # \c particles.0000000000.xml
    #
    # By default, only particle positions are output to the dump files. This can be changed
    # with set_params(), or by specifying the options in the dump.xml() command.
    #
    # If \a period is not specified, then no periodic updates will occur. Instead, the file
    # \a filename is written immediately.
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename="dump", period=None, **params):
        util.print_status_line();
    
        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.HOOMDDumpWriter(globals.system_definition, filename);
        util._disable_status_lines = True;
        self.set_params(**params);
        util._disable_status_lines = False;
        
        if period is not None:
            self.setupAnalyzer(period);
            self.enabled = True;
            self.prev_period = 1;
        elif filename != "dump":
            util._disable_status_lines = True;
            self.write(filename);
            util._disable_status_lines = False;
        else:
            self.enabled = False;

    ## Change xml write parameters
    #
    # \param all (if True) Enables the output of all optional parameters below
    # \param vis (if True) Enables options commonly used for visualization.
    # - Specifically, vis=True sets position, mass, diameter, type, body, bond, angle, dihedral, improper, charge
    # \param position (if set) Set to True/False to enable/disable the output of particle positions in the xml file
    # \param image (if set) Set to True/False to enable/disable the output of particle images in the xml file
    # \param velocity (if set) Set to True/False to enable/disable the output of particle velocities in the xml file
    # \param mass (if set) Set to True/False to enable/disable the output of particle masses in the xml file
    # \param diameter (if set) Set to True/False to enable/disable the output of particle diameters in the xml file
    # \param type (if set) Set to True/False to enable/disable the output of particle types in the xml file
    # \param body (if set) Set to True/False to enable/disable the output of the particle bodies in the xml file
    # \param wall (if set) Set to True/False to enable/disable the output of walls in the xml file
    # \param bond (if set) Set to True/False to enable/disable the output of bonds in the xml file
    # \param angle (if set) Set to True/False to enable/disable the output of angles in the xml file
    # \param dihedral (if set) Set to True/False to enable/disable the output of dihedrals in the xml file
    # \param improper (if set) Set to True/False to enable/disable the output of impropers in the xml file
    # \param acceleration (if set) Set to True/False to enable/disable the output of particle accelerations in the xml 
    # \param charge (if set) Set to True/False to enable/disable the output of particle charge in the xml 
    # \param orientation (if set) Set to True/False to enable/disable the output of paritle orientations in the xml file
    # \param vizsigma (if set) Set to a floating point value to include as vizsigma in the xml file
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
    def set_params(self,
                   all=None,
                   vis=None,
                   position=None,
                   image=None,
                   velocity=None,
                   mass=None,
                   diameter=None,
                   type=None,
                   body=None,
                   wall=None,
                   bond=None,
                   angle=None,
                   dihedral=None,
                   improper=None,
                   acceleration=None,
                   charge=None,
                   orientation=None,
                   vizsigma=None):
        util.print_status_line();
        self.check_initialization();
        
        if all:
            position = image = velocity = mass = diameter = type = wall = bond = angle = dihedral = improper = True;
            acceleration = charge = body = orientation = True;

        if vis:
            position = mass = diameter = type = body = bond = angle = dihedral = improper = charge = True;

        if position is not None:
            self.cpp_analyzer.setOutputPosition(position);

        if image is not None:
            self.cpp_analyzer.setOutputImage(image);

        if velocity is not None:
            self.cpp_analyzer.setOutputVelocity(velocity);
            
        if mass is not None:
            self.cpp_analyzer.setOutputMass(mass);
            
        if diameter is not None:
            self.cpp_analyzer.setOutputDiameter(diameter);
            
        if type is not None:
            self.cpp_analyzer.setOutputType(type);
        
        if body is not None:
            self.cpp_analyzer.setOutputBody(body);
        
        if wall is not None:
            self.cpp_analyzer.setOutputWall(wall);
            
        if bond is not None:
            self.cpp_analyzer.setOutputBond(bond);

        if angle is not None:
            self.cpp_analyzer.setOutputAngle(angle);
            
        if dihedral is not None:
            self.cpp_analyzer.setOutputDihedral(dihedral);
            
        if improper is not None:
            self.cpp_analyzer.setOutputImproper(improper);
        
        if acceleration is not None:
            self.cpp_analyzer.setOutputAccel(acceleration);
            
        if charge is not None:
            self.cpp_analyzer.setOutputCharge(charge);
        
        if orientation is not None:
            self.cpp_analyzer.setOutputOrientation(orientation);

        if vizsigma is not None:
            self.cpp_analyzer.setVizSigma(vizsigma);
        
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
        self.check_initialization();
        
        self.cpp_analyzer.writeFile(filename, globals.system.getCurrentTimeStep());

## Writes simulation snapshots in a binary format
#
# Every \a period time steps, a new file will be created. The state of the 
# particles at that time step is written to the file in a binary format.
#
# \sa init.read_bin
# \MPI_NOT_SUPPORTED
class bin(analyze._analyzer):
    ## Initialize the hoomd_bin writer
    #
    # \param filename (optional) Base of the file name
    # \param period (optional) Number of time steps between file dumps
    # \param file1 (optional) First alternating file name to write
    # \param file2 (optional) Second alternating file name to write
    # \param compress Set to False to disable gzip compression
    # 
    # \b Examples:
    # \code
    # dump.bin(file1="restart.1.bin.gz", file2="restart.2.bin.gz", period=1e5)
    # dump.bin(filename="particles", period=1000)
    # bin = dump.bin(filename="particles", period=1e5, compress=False)
    # bin = dump.bin()
    # \endcode
    #
    # If period is set, a new file will be created every \a period steps. The time step at which 
    # the file is created is added to the file name in a fixed width format to allow files to easily 
    # be read in order. I.e. the write at time step 0 with \c filename="particles" produces the file 
    # \c particles.0000000000.bin (.gz if \a compress = True)
    #
    # If \a compress is True (the default), output will be gzip compressed for a significant savings. init.read_bin()
    # will auto-detect whether or not the %data needs to be decompressed by the ".gz" file extension.
    #
    # If \a file1 and \a file2 are specified, then the output is written every \a period time steps alternating
    # between those two files. This use-case is useful when only the most recent state of the system is needed
    # to continue a job. The alternation between two files is so that if the job ends or crashes while writing one of
    # of the files, the other is still available for use. Make sure to include a .gz file extension if compression
    # is enabled.
    #
    # Binary files include the \b entire state of the system, including the time step, particle positions,
    # velocities, et cetera, and the internal state variables of any relevant integration methods. All %data is saved
    # exactly as it appears in memory so that loading the %data with init.read_bin is as close as possible as one
    # can get to exactly restarting the simulation as if it has never stopped. Restarts that continue 100%
    # \b exactly, bit for bit, as they would have if they had never stopped are possible under certain circumstances.
    #
    # For the integration state information to be read and properly associated with the correct integrator, there must
    # be the same number of integrator modes and methods <b>and in the same order</b> in the continuation script as are
    # in the initial job script. One way to ensure this is to copy the initial script and comment out all of the run()
    # commands up until the point that the restart file was written. Another method is to utilize run_upto() in your
    # job script.
    #
    # If \a period is not specified, then no periodic updates will occur. Instead, the file
    # \a filename is written immediately.
    #
    # \note The binary file format may change from one hoomd release to the next. Efforts will be made so that
    # newer versions of hoomd can read previous version's binary format, but support is not guaranteed. The intended
    # use case for dump.bin() is for saving data to restart and/or continue jobs that fail or reach a %wall clock time
    # limit. If you need to store data in a system and version independent manner, use dump.xml().
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename="dump", period=None, file1=None, file2=None, compress=True):
        util.print_status_line();
  
        # Error out in MPI simulations
        if (hoomd.is_MPI_available()):
            if globals.system_definition.getParticleData().getDomainDecomposition():
                globals.msg.error("dump.bin is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error writing restart data.")

        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.HOOMDBinaryDumpWriter(globals.system_definition, filename);
        self.cpp_analyzer.enableCompression(compress)
        
        # handle the alternation setting
        # first, check that they are both set
        if (file1 is not None and file2 is None) or (file2 is not None and file1 is None):
            globals.msg.error("file1 and file2 must either both be set or both left as None.\n");
            raise RuntimeError('Error initializing dump.bin');
        if file1 is not None:
            self.cpp_analyzer.setAlternatingWrites(file1, file2)
            if period is None:
                globals.msg.warning("Alternating file output set for dump.bin, but period is not set.\n");
                globals.msg.warning("No output will be written.\n");
        
        globals.msg.warning("dump.bin does not support triclinic boxes.\n");
        globals.msg.warning("dump.bin is deprecated and will be replaced in v1.1.0\n");
        
        if period is not None:
            self.setupAnalyzer(period);
            self.enabled = True;
            self.prev_period = 1;
        elif filename != "dump":
            util._disable_status_lines = True;
            self.write(filename);
            util._disable_status_lines = False;
        else:
            self.enabled = False;

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
        self.check_initialization();
        
        self.cpp_analyzer.writeFile(filename, globals.system.getCurrentTimeStep());

## Writes a simulation snapshot in the MOL2 format
#
# Every \a period time steps, a new file will be created. The state of the 
# particles at that time step is written to the file in the MOL2 format.
#
# Particle positions are written directly in distance units, see \ref page_units for more information.
#
# The intended usage is to use write() to generate a single structure file that 
# can be used by VMD for reading in particle names and %bond topology Use in 
# conjunction with dump.dcd for reading the full simulation trajectory into VMD.
# \MPI_NOT_SUPPORTED
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
    # If \a period is not specified, then no periodic updates will occur. Instead, the file
    # \a filename is written immediately.
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename="dump", period=None):
        util.print_status_line();

        # Error out in MPI simulations
        if (hoomd.is_MPI_available()):
            if globals.system_definition.getParticleData().getDomainDecomposition():
                globals.msg.error("dump.mol2 is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error writing MOL2 file.")

        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.MOL2DumpWriter(globals.system_definition, filename);
        
        if period is not None:
            self.setupAnalyzer(period);
            self.enabled = True;
            self.prev_period = 1;
        elif filename != "dump":
            util._disable_status_lines = True;
            self.write(filename);
            util._disable_status_lines = False;
        else:
            self.enabled = False;
                
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
        self.check_initialization();
        
        self.cpp_analyzer.writeFile(filename);

    
## Writes simulation snapshots in the DCD format
#
# Every \a period time steps a new simulation snapshot is written to the 
# specified file in the DCD file format. DCD only stores particle positions
# but is decently space efficient and extremely fast to read and write. VMD
# can load 100's of MiB of trajectory data in mere seconds.
#
# Particle positions are written directly in distance units, see \ref page_units for more information.
#
# Use in conjunction with dump.xml so that VMD has information on the
# particle names and %bond topology.
#
# Due to constraints of the DCD file format, once you stop writing to
# a file via disable(), you cannot continue writing to the same file,
# nor can you change the period of the %dump at any time. Either of these tasks 
# can be performed by creating a new %dump file with the needed settings.
#
# \MPI_SUPPORTED
class dcd(analyze._analyzer):
    ## Initialize the dcd writer
    #
    # \param filename File name to write
    # \param period Number of time steps between file dumps
    # \param group Particle group to output to the dcd file. If left as None, all particles will be written
    # \param overwrite When False, (the default) an existing DCD file will be appended to. When True, an existing DCD
    #        file \a filename will be overwritten.
    # \param unwrap_full When False, (the default) particle coordinates are always written inside the simulation box.
    #        When True, particles will be unwrapped into their current box image before writing to the dcd file.
    # \param unwrap_rigid When False, (the default) individual particles are written inside the simulation box which
    #        breaks up rigid bodies near box boundaries. When True, particles belonging to the same rigid body will be
    #        unwrapped so that the body is continuous. The center of mass of the body remains in the simulation box, but
    #        some particles may be written just outside it. \a unwrap_rigid is ignored if \a unwrap_full is True.
    # 
    # \b Examples:
    # \code
    # dump.dcd(filename="trajectory.dcd", period=1000)
    # dcd = dump.dcd(filename"data/dump.dcd", period=1000)
    # \endcode
    #
    # \warning 
    # When you use dump.dcd to append to an existing dcd file
    # - The period must be the same or the time data in the file will not be consistent.
    # - dump.dcd will not write out data at time steps that already are present in the dcd file to maintain a
    #   consistent timeline
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename, period, group=None, overwrite=False, unwrap_full=False, unwrap_rigid=False):
        util.print_status_line();
        
        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        reported_period = period;
        if type(period) != type(1):
            reported_period = 1000;
            
        if group is None:
            util._disable_status_lines = True;
            group = hs_group.all();
            util._disable_status_lines = False;
            
        self.cpp_analyzer = hoomd.DCDDumpWriter(globals.system_definition, filename, int(reported_period), group.cpp_group, overwrite);
        self.cpp_analyzer.setUnwrapFull(unwrap_full);
        self.cpp_analyzer.setUnwrapRigid(unwrap_rigid);
        self.setupAnalyzer(period);
    
    def enable(self):
        util.print_status_line();
        
        if self.enabled == False:
            globals.msg.error("you cannot re-enable DCD output after it has been disabled\n");
            raise RuntimeError('Error enabling updater');
    
    def set_period(self, period):
        util.print_status_line();
        
        globals.msg.error("you cannot change the period of a dcd dump writer\n");
        raise RuntimeError('Error changing updater period');


## Writes simulation snapshots in the PBD format
#
# Every \a period time steps, a new file will be created. The state of the 
# particles at that time step is written to the file in the PDB format.
#
# Particle positions are written directly in distance units, see \ref page_units for more information.
# \MPI_NOT_SUPPORTED
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
    # If \a period is not specified, then no periodic updates will occur. Instead, the file
    # \a filename is written immediately.
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename="dump", period=None):
        util.print_status_line();

        # Error out in MPI simulations
        if (hoomd.is_MPI_available()):
            if globals.system_definition.getParticleData().getDomainDecomposition():
                globals.msg.error("dump.pdb is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error writing PDB file.")


        # initialize base class
        analyze._analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.PDBDumpWriter(globals.system_definition, filename);
        
        if period is not None:
            self.setupAnalyzer(period);
            self.enabled = True;
            self.prev_period = 1;
        elif filename != "dump":
            util._disable_status_lines = True;
            self.write(filename);
            util._disable_status_lines = False;
        else:
            self.enabled = False;
    
    ## Change pdb write parameters
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
        self.check_initialization();
            
        if bond is not None:
            self.cpp_analyzer.setOutputBond(bond);
    
    ## Write a file at the current time step
    #
    # \param filename File name to write
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
        self.check_initialization();
        
        self.cpp_analyzer.writeFile(filename);

