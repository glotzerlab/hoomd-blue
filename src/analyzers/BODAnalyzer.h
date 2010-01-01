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

// $Id: BODAnalyzer.h 2148 2009-10-07 20:05:29Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/src/analyzers/BODAnalyzer.h $
// Maintainer: joaander

/*! \file BODAnalyzer.h
    \brief Declares the BODAnalyzer class
*/

#include <string>
#include <fstream>
#include <boost/shared_ptr.hpp>

#include "Analyzer.h"
#include "ParticleGroup.h"

#ifndef __BOD_ANALYZER_H__
#define __BOD_ANALYZER_H__

//! Prints a log of a bond order diagram calculated over particles in the simulation
/*! At each time step specified, BODAnalyzer opens a given file name (overwriting it if it exists) for writing.  The output
    contains a list of (normalized) x, y, z, vectors or r, theta, rho vectors (as determined by the user) of the bond order diagram.
    Each time analyze() is called, the structure factor is 
    calculated and written out to the file.

   The Bond Order Diagram will be calculated for particles of group A, with respect to neighbors from group B, which may be the same
   group.

    \ingroup analyzers
*/
class BODAnalyzer : public Analyzer
    {
    public:
        //! Construct the BOD analyzer
        BODAnalyzer(boost::shared_ptr<SystemDefinition> sysdef, const std::string, boost::shared_ptr<ParticleGroup> groupA, boost::shared_ptr<ParticleGroup> groupB, Scalar cutoff);
        
        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);
        
        //! Sets the delimiter to use between fields
        void setDelimiter(const std::string& delimiter);
        
        //! Changes the coordinates output from xyz to spherical
        void setCoordinatesSpherical() {m_usesphericalcoords=true;}
        
    private:
        //! The delimiter to put between columns in the file
        std::string m_delimiter;

        //! The base name of the file
        std::string m_filename;  
        
        boost::shared_ptr<ParticleGroup const> m_groupA; //!< A shared pointer to the group A definition
        boost::shared_ptr<ParticleGroup const> m_groupB; //!< A shared pointer to the group B definition
        
        Scalar m_cutoffsq;  //!<  The cutoff used to determine neighbors for the bond-order diagram.
        
        bool m_usesphericalcoords;
        
        //! Helper function to calculate the BOD of groupA with respect to group B
        void calcBOD(ofstream &m_file);
        //! Helper function to write one row of output
        void writeFile(unsigned int timestep);
    };

//! Exports the BODAnalyzer class to python
void export_BODAnalyzer();

#endif

