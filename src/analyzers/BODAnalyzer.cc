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

// $Id: BODAnalyzer.cc 2148 2009-10-07 20:05:29Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/src/analyzers/BODAnalyzer.cc $
// Maintainer: phillicl

/*! \file BODAnalyzer.cc
    \brief Defines the BODAnalyzer class
*/

#include "BODAnalyzer.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <iomanip>
using namespace std;

/*! \param sysdef SystemDefinition containing the Particle data to analyze

*/
BODAnalyzer::BODAnalyzer(boost::shared_ptr<SystemDefinition> sysdef,
            const std::string fname,
            boost::shared_ptr<ParticleGroup> groupA, 
            boost::shared_ptr<ParticleGroup> groupB, 
            Scalar cutoffsq)
    : Analyzer(sysdef), m_delimiter("\t"), m_filename(fname), m_groupA(groupA), m_groupB(groupB), m_cutoffsq(cutoffsq), m_usesphericalcoords(false)
    {
    }

/*!\param timestep Current time step of the simulation

    analyze() will first write out the file header

    On every call, analyze() will write calculate the BOD for groupA with respect to groupB, and generate a file of coordinates.
*/
void BODAnalyzer::analyze(unsigned int timestep)
    {
        
    // Write out a single file containing the bond-order-diagram at that timestep. 
        
    // write out the new file
    writeFile(timestep);
        
    }

/*! \param delimiter New delimiter to set

    The delimiter is printed between every element in the row of the output
*/
void BODAnalyzer::setDelimiter(const std::string& delimiter)
    {
    m_delimiter = delimiter;
    }

/*! \param group Particle group to calculate the BOD of
    Calculates whether given particles A and B are neighbors.  Currently does this by a brute force calculation, which is why if the BOD is calculated frequently,
    will significantly slow the simulation. 
*/
void BODAnalyzer::calcBOD(ofstream &m_file)
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    BoxDim box = m_pdata->getBox();
    Scalar Lx = (box.xhi - box.xlo);
    Scalar Ly = (box.yhi - box.ylo);
    Scalar Lz = (box.zhi - box.zlo);
    Scalar dx, dy, dz;
    Scalar invnorm;
    Scalar nx, ny, nz;
    Scalar rho, phi, theta;
    rho=1.0;
    
    // for each particle in the group

    for (unsigned int group_idA = 0; group_idA < m_groupA->getNumMembers(); group_idA++)
        {
        // get the tag for the current group member from the group
        unsigned int tagA = m_groupA->getMemberTag(group_idA);
        // identify the index of the current particle tag
        unsigned int idA = arrays.rtag[tagA];    
       
        for (unsigned int group_idB = 0; group_idB < m_groupB->getNumMembers(); group_idB++)
            {         
    
            // get the tag for the current group member from the group
            unsigned int tagB = m_groupB->getMemberTag(group_idB);
            // identify the index of the current particle tag
            unsigned int idB = arrays.rtag[tagB];  
            
            if (idA != idB)  //carefully rule out this case for when the groups have overlap.
                    {
                    dx = arrays.x[idB]-arrays.x[idA];
                    if (dx > Lx/2.0)   dx -= Lx;
                    if (dx < -Lx/2.0)  dx += Lx;                                
                    dy = arrays.y[idB]-arrays.y[idA];
                    if (dy > Ly/2.0)   dy -= Ly;
                    if (dy < -Ly/2.0)  dy += Ly;                    
                    dz = arrays.z[idB]-arrays.z[idA];
                    if (dz > Lz/2.0)   dz -= Lz;
                    if (dz < -Lz/2.0)  dz += Lz;
                    
                    if (dx*dx + dy*dy + dz*dz <= m_cutoffsq) 
                      {
                      invnorm = 1.0/sqrt(dx*dx + dy*dy + dz*dz);
                      nx = dx*invnorm;
                      ny = dy*invnorm; 
                      nz = dz*invnorm;
                      
                      if (m_usesphericalcoords) 
                        {
                        phi = acos(nz);
                        if (dx >= 0)  theta = asin(ny/sqrt(nx*nx + ny*ny));
                        else          theta = M_PI - asin(ny/sqrt(nx*nx + ny*ny));
                        m_file << rho << m_delimiter << phi << m_delimiter << theta << endl;
                        }
                      else
                        m_file << nx << m_delimiter << ny << m_delimiter << nz << endl;
                      }                                          
                    }
            }
        }   
                        
    m_pdata->release();
    return;
    }

/*! \param timestep current time step of the simulation

    Performs all the steps needed in order to calculate the BODs and write the values to a file.
*/
void BODAnalyzer::writeFile(unsigned int timestep)
    {
    if (m_prof) m_prof->push("BOD");
       
        
    //create the file for this time step
    ostringstream full_fname;
    string filetype = ".BOD";
    
    // Generate a filename with the timestep padded to ten zeros
    full_fname << m_filename << "." << setfill('0') << setw(10) << timestep << filetype;        
    // open the file for writing
    ofstream m_file(full_fname.str().c_str());    
  
    if (!m_file.good())
       {
       cerr << endl << "***Error! Unexpected error writing BOD file" << endl << endl;
       throw runtime_error("Error writting BOD file");
       }            
    
    if (m_usesphericalcoords)
        m_file << "rho" << m_delimiter << "phi" << m_delimiter << "theta" << endl;
    else     
        m_file << "x" << m_delimiter << "y" << m_delimiter << "z" << endl;
        
    //Print Each Line 
    calcBOD(m_file);    
 
    m_file.close();  //Close the file
        
    if (m_prof) m_prof->pop();
    }

void export_BODAnalyzer()
    {
    class_<BODAnalyzer, boost::shared_ptr<BODAnalyzer>, bases<Analyzer>, boost::noncopyable>
    ("BODAnalyzer", init< boost::shared_ptr<SystemDefinition>, 
                    const std::string&, 
                    boost::shared_ptr<ParticleGroup>, 
                    boost::shared_ptr<ParticleGroup>, 
                    Scalar
                    >())
    .def("setDelimiter", &BODAnalyzer::setDelimiter)
    .def("setCoordinatesSpherical", &BODAnalyzer::setCoordinatesSpherical)
    ;
    }

