/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file MSDAnalyzer.h
    \brief Declares the MSDAnalyzer class
*/

#include <string>
#include <fstream>
#include <boost/shared_ptr.hpp>

#include "Analyzer.h"
#include "ParticleGroup.h"

#ifndef __MSD_ANALYZER_H__
#define __MSD_ANALYZER_H__

//! Prints a log of the mean-squared displacement calculated over particles in the simulation
/*! On construction, MSDAnalyzer opens the given file name (overwriting it if it exists) for writing. It also records
    the initial positions of all particles in the simulation. Each time analyze() is called, the mean-squared
    displacement is calculated and written out to the file.

    The mean squared displacement (MSD) is calculated as:
    \f[ \langle |\vec{r} - \vec{r}_0|^2 \rangle \f]

    Multiple MSD columns may be desired in a single simulation run. Rather than requiring the user to specify
    many analyze.msd commands each with a separate file, a single class instance is designed to be capable of outputting
    many columns. The particles over which the MSD is calculated for each column are specified with a ParticleGroup.

    \ingroup analyzers
*/
class MSDAnalyzer : public Analyzer
    {
    public:
        //! Construct the msd analyzer
        MSDAnalyzer(boost::shared_ptr<SystemDefinition> sysdef, std::string fname, const std::string& header_prefix="");
        
        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);
        
        //! Sets the delimiter to use between fields
        void setDelimiter(const std::string& delimiter);
        
        //! Adds a column to the analysis
        void addColumn(boost::shared_ptr<ParticleGroup> group, const std::string& name);
        
    private:
        //! The delimiter to put between columns in the file
        std::string m_delimiter;
        //! The prefix written at the beginning of the header line
        std::string m_header_prefix;
        
        bool m_columns_changed; //!< Set to true if the list of columns have changed
        std::ofstream m_file;   //!< The file we write out to
        
        std::vector<Scalar> m_initial_x;    //!< initial value of the x-component listed by tag
        std::vector<Scalar> m_initial_y;    //!< initial value of the y-component listed by tag
        std::vector<Scalar> m_initial_z;    //!< initial value of the z-component listed by tag
        
        //! struct for storing the particle group and name assocated with a column in the output
        struct column
            {
            //! default constructor
            column() {}
            //! constructs a column
            column(boost::shared_ptr<ParticleGroup const> group, const std::string& name) :
                    m_group(group), m_name(name) {}
                    
            boost::shared_ptr<ParticleGroup const> m_group; //!< A shared pointer to the group definition
            std::string m_name;                             //!< The name to print across the file header
            };
            
        std::vector<column> m_columns;  //!< List of groups to output
        
        //! Helper function to write out the header
        void writeHeader();
        //! Helper function to calculate the MSD of a single group
        Scalar calcMSD(boost::shared_ptr<ParticleGroup const> group);
        //! Helper function to write one row of output
        void writeRow(unsigned int timestep);
    };

//! Exports the MSDAnalyzer class to python
void export_MSDAnalyzer();

#endif
