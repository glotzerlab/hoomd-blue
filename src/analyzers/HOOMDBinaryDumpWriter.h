/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file HOOMDBinaryDumpWriter.h
    \brief Declares the HOOMDBinaryDumpWriter class
*/

#include <string>

#include <boost/shared_ptr.hpp>

#include "Analyzer.h"

#ifndef __HOOMD_BINARY_DUMP_WRITER_H__
#define __HOOMD_BINARY_DUMP_WRITER_H__

//! Analyzer for writing out HOOMD  dump files
/*! HOOMDBinaryDumpWriter can be used to write out xml files containing various levels of information
    of the current time step of the simulation. At a minimum, the current time step and box
    dimensions are output. Optionally, particle positions, velocities and types can be included
    in the file.

    Usage:<br>
    Construct a HOOMDBinaryDumpWriter, attaching it to a ParticleData and specifying a base file name.
    Call analyze(timestep) to output a dump file with the state of the current time step
    of the simulation. It will create base_file.timestep.xml where timestep is a 0-padded
    10 digit number. The 0 padding is so files sorted "alphabetically" will be read in
    numberical order.

    To include positions, velocities and types, see: setOutputPosition() setOutputVelocity()
    and setOutputType(). Similarly, walls and bonds can be included with setOutputWall() and
    setOutputBond().

    Future versions will include the ability to dump forces on each particle to the file also.

    For information on the structure of the xml file format: see \ref page_dev_info
    Although, HOOMD's  user guide probably has a more up to date documentation on the format.
    \ingroup analyzers
*/
class HOOMDBinaryDumpWriter : public Analyzer
    {
    public:
        //! Construct the writer
        HOOMDBinaryDumpWriter(boost::shared_ptr<SystemDefinition> sysdef, std::string base_fname);
        
        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);        
        //! Writes a file at the current time step
        void writeFile(std::string fname, unsigned int timestep);
        //! Set the alternating mode
        void setAlternatingWrites(const std::string& fname1, const std::string& fname2);
        //! Enable or disable gzip compression of the binary output files
        void enableCompression(bool enable_compression);
    private:
        std::string m_base_fname;   //!< String used to store the file name of the XML file
        std::string m_fname1;       //!< File name for the first file to write to in alternating mode
        std::string m_fname2;       //!< File name for the second file to write to in alternating mode
        bool m_alternating;         //!< True if we are to write to m_fname1 and m_fname in an alternating fasion
        unsigned int m_cur_file;    //!< Current index of the file we are writing to (1 or 2)
        bool m_enable_compression;  //!< True if gzip compression should be enabled
        };

//! Exports the HOOMDBinaryDumpWriter class to python
void export_HOOMDBinaryDumpWriter();

#endif

