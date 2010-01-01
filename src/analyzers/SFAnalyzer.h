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

// $Id: SFAnalyzer.h 2148 2009-10-07 20:05:29Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/src/analyzers/SFAnalyzer.h $
// Maintainer: joaander

/*! \file SFAnalyzer.h
    \brief Declares the SFAnalyzer class
*/

#include <string>
#include <fstream>
#include <boost/shared_ptr.hpp>

#include "Analyzer.h"
#include "ParticleGroup.h"

#ifndef __SF_ANALYZER_H__
#define __SF_ANALYZER_H__

//! Prints a the structure factor calculated over particles in the simulation
/*! At each specified time step, SFAnalyzer opens the given file name (overwriting it if it exists) for writing.  The first column
    conrresponds to the magnitude of the q vector.  The second column corresponds to the magnitude of the m vector.
    Each time analyze() is called, the structure factor is 
    calculated and written out to the file.

    The Structure Factor (SF) is calculated as:
    \f[ S(\vec{q}) = \frac{ (\sum_j cos(\vec{q}\cdot\vec{r}_j))^2 + (\sum_j cos(\vec{q}\cdot\vec{r}_j))^2 }{N} \f]
    \f[ |\vec{q}| = 2\pi\sqrt{\frac{i^2}{L_x} + \frac{j^2}{L_y} + \frac{k^2}{L_z}} \f]
    \f[ |\vec{m}| = \sqrt{i^2 + j^2 + k^2} \f]

    Multiple SF groups may be desired in a single simulation run. Rather than requiring the user to specify
    many analyze.sf commands, a single class instance is designed to be capable of outputting
    many groups. The particles over which the SF is calculated for each group are specified with a ParticleGroup.

    \ingroup analyzers
*/
class SFAnalyzer : public Analyzer
    {
    public:
        //! Construct the sf analyzer
        SFAnalyzer(boost::shared_ptr<SystemDefinition> sysdef, const std::string);
        
        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);
        
        //! Sets the delimiter to use between fields
        void setDelimiter(const std::string& delimiter);
        
        //! Adds a column to the analysis
        void addGroup(boost::shared_ptr<ParticleGroup> group, const std::string& gname, int vec_div);
        
        //! Turns on show components
        void showComponents() {m_showcomponents=true;} 
        
    private:
        //! The delimiter to put between columns in the file
        std::string m_delimiter;

        //! The base name of the file
        std::string m_filename;  
        
        unsigned int m_maxi;
        unsigned int m_maxnum_rows;
        bool m_showcomponents;
        
        bool m_SFgroups_changed; //!< Set to true if the list of SFgroups have changed
        
        //! struct for storing the particle group and name assocated with a column in the output
        struct SFgroup
            {
            //! default constructor
            SFgroup() {}
            //! constructs a column
            SFgroup(boost::shared_ptr<ParticleGroup const> group, const std::string& gname, unsigned int vec_div) :
                    m_group(group), m_gname(gname), m_vec_div(vec_div) {}
                    
            boost::shared_ptr<ParticleGroup const> m_group; //!< A shared pointer to the group definition

            //To Add Later - Let user specify if they want the full vector, not just magnitude.
            std::vector<Scalar> m_q;    //!< magnitude of q vector
            std::vector<Scalar> m_qi;    //!< xcomponent of q vector
            std::vector<Scalar> m_qj;    //!< ycomponent of q vector
            std::vector<Scalar> m_qk;    //!< zcomponent of q vector            
            std::vector<Scalar> m_Sq;   //!< Structure Factor vector   

           // std::ofstream m_file;   //!< The file we write out to
            std::string m_gname;      //!< The base file name we want to write to
            
            //! The limit of i, j, and k, or how many wavelengths the box is split into.
            unsigned int m_vec_div;                  
            
            };
            
        std::vector<SFgroup> m_SFgroups;  //!< List of groups to output

        
        //! Helper function to calculate the SF of a single group
        Scalar calcSF(SFgroup & sfgroup);
        //! Helper function to write one row of output
        void writeFile(unsigned int timestep);
    };

//! Exports the SFAnalyzer class to python
void export_SFAnalyzer();

#endif

