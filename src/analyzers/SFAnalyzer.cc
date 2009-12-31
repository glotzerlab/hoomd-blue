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

// $Id: SFAnalyzer.cc 2148 2009-10-07 20:05:29Z joaander $
// $URL: https://codeblue.umich.edu/hoomd-blue/svn/trunk/src/analyzers/SFAnalyzer.cc $
// Maintainer: phillicl

/*! \file SFAnalyzer.cc
    \brief Defines the SFAnalyzer class
*/

#include "SFAnalyzer.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <iomanip>
using namespace std;

/*! \param sysdef SystemDefinition containing the Particle data to analyze

*/
SFAnalyzer::SFAnalyzer(boost::shared_ptr<SystemDefinition> sysdef,
            const std::string fname)
    : Analyzer(sysdef), m_delimiter("\t"), m_filename(fname), m_maxi(0), m_maxnum_rows(0), m_SFgroups_changed(false)
    {
    }

/*!\param timestep Current time step of the simulation

    analyze() will first write out the file header if the columns have changed.

    On every call, analyze() will write calculate the SF for each group and write out a row in the file.
*/
void SFAnalyzer::analyze(unsigned int timestep)
    {
    // error check
    if (m_SFgroups.size() == 0)
        {
        cout << "***Warning! No groups specified in the SF analysis" << endl;
        return;
        }
        
    // For each SFgroup, write out a single file containing the structure factor at that timestep. 
    // To do - Add ability to average structure factor.  But need to determine how SF are aveaged (by including all particles)
    // or by averaging the values at each index.
    // write out the header only once if the columns change
        
    // write out the new file
    writeFile(timestep);
        
    }

/*! \param delimiter New delimiter to set

    The delimiter is printed between every element in the row of the output
*/
void SFAnalyzer::setDelimiter(const std::string& delimiter)
    {
    m_delimiter = delimiter;
    }

/*! \param group Particle group to calculate the SF of
    \param gname column name
    \param vec_div The maximum division of the box for calculating the structure factor.
    \param header_prefix String to print before the file header

    On construction, the q and m values are recorded, for the given vector divisions. The file is opened
    (and overwritten if necessary). Nothing is initially written to the file, that will occur on the first call to
    analyze()    

    After a group is added with addGroup(), future calls to analyze() will calculate the SF of the particles defined
    in \a group and print out an entry under the \a name header in the file.
*/
void SFAnalyzer::addGroup(boost::shared_ptr<ParticleGroup> group,
                          const std::string& gname,
                          int vec_div)
    {
    m_SFgroups.push_back(SFgroup(group, gname, vec_div));

    unsigned int current = m_SFgroups.size() - 1;
    
    unsigned int mylength = (vec_div + 1)*(vec_div + 1)*(vec_div + 1) - 1;
    if (mylength > m_maxnum_rows) 
        {
        m_maxnum_rows = mylength;
        m_maxi = current; 
        }

     // initialize the number of q and m magnitudes to mylength       
    m_SFgroups[current].m_q.resize(mylength);
    m_SFgroups[current].m_m.resize(mylength);
    m_SFgroups[current].m_mi.resize(mylength);
    m_SFgroups[current].m_mj.resize(mylength);
    m_SFgroups[current].m_mk.resize(mylength);    
    m_SFgroups[current].m_Sq.resize(mylength);
    
    BoxDim box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    unsigned int q_index = 0;
    // Initialize the values
    for (unsigned int ivec = 0; ivec <= vec_div; ivec++)
        {
        for (unsigned int jvec = 0; jvec <= vec_div; jvec++)
            {
            for (unsigned int kvec = 0; kvec <= vec_div; kvec++)
                {    
                if (ivec !=0 | jvec != 0 | kvec !=0) 
                    {    
                    q_index = ivec + (vec_div+1)*jvec + (vec_div+1)*(vec_div+1)*kvec;
                    m_SFgroups[current].m_q[q_index-1] = 2*M_PI*sqrt((Scalar) ivec*ivec/(Lx*Lx) + (Scalar) jvec*jvec/(Ly*Ly) + (Scalar) kvec*kvec/(Lz*Lz));
                    m_SFgroups[current].m_m[q_index-1] = sqrt(ivec*ivec + jvec*jvec + kvec*kvec);
                    m_SFgroups[current].m_mi[q_index-1] = ivec;
                    m_SFgroups[current].m_mj[q_index-1] = jvec;
                    m_SFgroups[current].m_mk[q_index-1] = kvec;                    
                    }
                }
            }
        }
    m_pdata->release();
    
    m_SFgroups_changed = true;
    }


/*! \param group Particle group to calculate the SF of
    Loop through all particles in the given group and calculate the SF over them.
*/
Scalar SFAnalyzer::calcSF(SFgroup & sfgroup)
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    BoxDim box = m_pdata->getBox();
    Scalar invLx = 1.0/(box.xhi - box.xlo);
    Scalar invLy = 1.0/(box.yhi - box.ylo);
    Scalar invLz = 1.0/(box.zhi - box.zlo);
    
    // initial sum
    Scalar Ssin = Scalar(0.0);
    Scalar Scos = Scalar(0.0);
    
    // handle the case where there are 0 members gracefully
    if (sfgroup.m_group->getNumMembers() == 0)
        {
        cout << "***Warning! Group has 0 members" << endl;
        return 1;
        }
    
    // get vec_div        
    unsigned int vec_div = sfgroup.m_vec_div;
    unsigned int q_index =0;
    // for each particle in the group
    for (unsigned int ivec = 0; ivec <= vec_div; ivec++)
        {
        for (unsigned int jvec = 0; jvec <= vec_div; jvec++)
            {
            for (unsigned int kvec = 0; kvec <= vec_div; kvec++)
                {    
                if (ivec !=0 | jvec != 0 | kvec !=0) 
                    {    
                    q_index = ivec + (vec_div + 1)*jvec + (vec_div + 1)*(vec_div + 1)*kvec;
                    Ssin = Scalar(0.0);
                    Scos = Scalar(0.0);
                    for (unsigned int group_idx = 0; group_idx < sfgroup.m_group->getNumMembers(); group_idx++)
                        {
                        // get the tag for the current group member from the group
                        unsigned int tag = sfgroup.m_group->getMemberTag(group_idx);
        
                        // identify the index of the current particle tag
                        unsigned int idx = arrays.rtag[tag];
                        
        
                        // save its initial position
                        Ssin += sin((ivec*arrays.x[idx]*invLx + jvec*arrays.y[idx]*invLy + kvec*arrays.z[idx]*invLz)*2*M_PI);
                        Scos += cos((ivec*arrays.x[idx]*invLx + jvec*arrays.y[idx]*invLy + kvec*arrays.z[idx]*invLz)*2*M_PI);
                        }
                    sfgroup.m_Sq[q_index - 1] = (Scalar) (Ssin*Ssin + Scos*Scos)/sfgroup.m_group->getNumMembers();
                    }
                }
            }
        }
        
    m_pdata->release();
    return 1    ;
    }

/*! \param timestep current time step of the simulation

    Performs all the steps needed in order to calculate the SFs for all the groups in the columns and writes out an
    entire row to the file.
*/
void SFAnalyzer::writeFile(unsigned int timestep)
    {
    if (m_prof) m_prof->push("SF");
       
    // quit now if there is nothing to log
    if (m_SFgroups.size() == 0)
        {
        return;
        }
        
    //create the file for this time step
    ostringstream full_fname;
    string filetype = ".sf";
    
    // Generate a filename with the timestep padded to ten zeros
    full_fname << m_filename << "." << setfill('0') << setw(10) << timestep << filetype;        
    // open the file for writing
    ofstream m_file(full_fname.str().c_str());    
  
    if (!m_file.good())
       {
       cerr << endl << "***Error! Unexpected error writing sf file" << endl << endl;
       throw runtime_error("Error writting sf file");
       }            
    
    // Format the x-axis as instructed.  (q is default)
    m_file << "q" << m_delimiter; 

    // write all but the last of the quantities separated by the delimiter
    for (unsigned int i = 0; i < m_SFgroups.size()-1; i++)
        m_file << m_SFgroups[i].m_gname << m_delimiter;
    // write the last one with no delimiter after it
    m_file << m_SFgroups[m_SFgroups.size()-1].m_gname << endl;
        
    // write files for all the groups
    for (unsigned int i = 0; i < m_SFgroups.size(); i++)  
        {
        //Calculate the Structure Factor for the Group
        calcSF(m_SFgroups[i]);
        }
                              
    //Print Each Line  (note, currently q versus S(q)... later will provide m option.
    for (unsigned int row = 0; row < m_maxnum_rows; row++) 
        {
        
        // Set to q as default, otherwise set to m
        m_file << setprecision(10) << m_SFgroups[m_maxi].m_m[row] << m_delimiter;
        m_file << m_SFgroups[m_maxi].m_mi[row] << m_delimiter << m_SFgroups[m_maxi].m_mj[row] << m_delimiter <<  m_SFgroups[m_maxi].m_mk[row] << m_delimiter;
       

        // write all but the last of the quantities separated by the delimiter
        for (unsigned int i = 0; i < m_SFgroups.size()-1; i++)
            if (row < m_SFgroups[i].m_q.size()) 
                m_file << m_SFgroups[i].m_Sq[row] << m_delimiter;
            else 
                m_file << 0 << m_delimiter;
        // write the last one with no delimiter after it
        if (row < m_SFgroups[m_SFgroups.size()-1].m_q.size()) 
            m_file << m_SFgroups[m_SFgroups.size()-1].m_Sq[row] << endl;
        else 
            m_file << 0 << endl;        
        
        }
    
    m_file.close();  //Close the file
        
    if (m_prof) m_prof->pop();
    }

void export_SFAnalyzer()
    {
    class_<SFAnalyzer, boost::shared_ptr<SFAnalyzer>, bases<Analyzer>, boost::noncopyable>
    ("SFAnalyzer", init< boost::shared_ptr<SystemDefinition>, const std::string& >())
    .def("setDelimiter", &SFAnalyzer::setDelimiter)
    .def("addGroup", &SFAnalyzer::addGroup)
    ;
    }

