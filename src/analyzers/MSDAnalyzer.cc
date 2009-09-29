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

/*! \file MSDAnalyzer.cc
    \brief Defines the MSDAnalyzer class
*/

#include "MSDAnalyzer.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <iomanip>
using namespace std;

/*! \param sysdef SystemDefinition containing the Particle data to analyze
    \param fname File name to write output to
    \param header_prefix String to print before the file header

    On construction, the initial coordinates of all parrticles in the system are recoreded. The file is opened
    (and overwritten if necessary). Nothing is initially written to the file, that will occur on the first call to
    analyze()
*/
MSDAnalyzer::MSDAnalyzer(boost::shared_ptr<SystemDefinition> sysdef, std::string fname, const std::string& header_prefix)
        : Analyzer(sysdef), m_delimiter("\t"), m_header_prefix(header_prefix), m_columns_changed(false), m_file(fname.c_str())
    {
    // record the initial particle positions by tag
    m_initial_x.resize(m_pdata->getN());
    m_initial_y.resize(m_pdata->getN());
    m_initial_z.resize(m_pdata->getN());
    
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    BoxDim box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // for each particle in the data
    for (unsigned int tag = 0; tag < arrays.nparticles; tag++)
        {
        // identify the index of the current particle tag
        unsigned int idx = arrays.rtag[tag];
        
        // save its initial position
        m_initial_x[tag] = arrays.x[idx] + Scalar(arrays.ix[idx]) * Lx;
        m_initial_y[tag] = arrays.y[idx] + Scalar(arrays.iy[idx]) * Ly;
        m_initial_z[tag] = arrays.z[idx] + Scalar(arrays.iz[idx]) * Lz;
        }
    m_pdata->release();
    }

/*!\param timestep Current time step of the simulation

    analyze() will first write out the file header if the columns have changed.

    On every call, analyze() will write calculate the MSD for each group and write out a row in the file.
*/
void MSDAnalyzer::analyze(unsigned int timestep)
    {
    // error check
    if (m_columns.size() == 0)
        {
        cout << "***Warning! No columns specified in the MSD analysis" << endl;
        return;
        }
        
    // write out the header only once if the columns change
    if (m_columns_changed)
        {
        writeHeader();
        m_columns_changed = false;
        }
        
    // write out the row every time
    writeRow(timestep);
    }

/*! \param delimiter New delimiter to set

    The delimiter is printed between every element in the row of the output
*/
void MSDAnalyzer::setDelimiter(const std::string& delimiter)
    {
    m_delimiter = delimiter;
    }

/*! \param group Particle group to calculate the MSD of
    \param name Name to print in the header of the file

    After a column is added with addColumn(), future calls to analyze() will calculate the MSD of the particles defined
    in \a group and print out an entry under the \a name header in the file.
*/
void MSDAnalyzer::addColumn(boost::shared_ptr<ParticleGroup> group, const std::string& name)
    {
    m_columns.push_back(column(group, name));
    m_columns_changed = true;
    }

/*! The entire header row is written to the file. First, timestep is written as every file includes it and then the
    columns are looped through and their names printed, separated by the delimiter.
*/
void MSDAnalyzer::writeHeader()
    {
    // write out the header prefix
    m_file << m_header_prefix;
    
    // timestep is always output
    m_file << "timestep";
    
    if (m_columns.size() == 0)
        {
        cout << "***Warning! No columns specified in the MSD analysis" << endl;
        return;
        }
        
    // only print the delimiter after the timestep if there are more columns
    m_file << m_delimiter;
    
    // write all but the last of the quantities separated by the delimiter
    for (unsigned int i = 0; i < m_columns.size()-1; i++)
        m_file << m_columns[i].m_name << m_delimiter;
    // write the last one with no delimiter after it
    m_file << m_columns[m_columns.size()-1].m_name << endl;
    m_file.flush();
    }

/*! \param group Particle group to calculate the MSD of
    Loop through all particles in the given group and calculate the MSD over them.
    \returns The calculated MSD
*/
Scalar MSDAnalyzer::calcMSD(boost::shared_ptr<ParticleGroup const> group)
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    BoxDim box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // initial sum for the average
    Scalar msd = Scalar(0.0);
    
    // handle the case where there are 0 members gracefully
    if (group->getNumMembers() == 0)
        {
        cout << "***Warning! Group has 0 members, reporting a calculated msd of 0.0" << endl;
        return Scalar(0.0);
        }
        
    // for each particle in the group
    for (unsigned int group_idx = 0; group_idx < group->getNumMembers(); group_idx++)
        {
        // get the tag for the current group member from the group
        unsigned int tag = group->getMemberTag(group_idx);
        
        // identify the index of the current particle tag
        unsigned int idx = arrays.rtag[tag];
        
        // save its initial position
        Scalar dx = arrays.x[idx] + Scalar(arrays.ix[idx]) * Lx - m_initial_x[tag];
        Scalar dy = arrays.y[idx] + Scalar(arrays.iy[idx]) * Ly - m_initial_y[tag];
        Scalar dz = arrays.z[idx] + Scalar(arrays.iz[idx]) * Lz - m_initial_z[tag];
        
        msd += dx*dx + dy*dy + dz*dz;
        }
    m_pdata->release();
    
    // divide to complete the average
    msd /= Scalar(group->getNumMembers());
    return msd;
    }

/*! \param timestep current time step of the simulation

    Performs all the steps needed in order to calculate the MSDs for all the groups in the columns and writes out an
    entire row to the file.
*/
void MSDAnalyzer::writeRow(unsigned int timestep)
    {
    if (m_prof) m_prof->push("MSD");
    
    // The timestep is always output
    m_file << setprecision(10) << timestep;
    
    // quit now if there is nothing to log
    if (m_columns.size() == 0)
        {
        return;
        }
        
    // only print the delimiter after the timestep if there are more columns
    m_file << m_delimiter;
    
    // write all but the last of the columns separated by the delimiter
    for (unsigned int i = 0; i < m_columns.size()-1; i++)
        m_file << setprecision(10) << calcMSD(m_columns[i].m_group) << m_delimiter;
    // write the last one with no delimiter after it
    m_file << setprecision(10) << calcMSD(m_columns[m_columns.size()-1].m_group) << endl;
    m_file.flush();
    
    if (!m_file.good())
        {
        cerr << endl << "***Error! Unexpected error writing msd file" << endl << endl;
        throw runtime_error("Error writting msd file");
        }
        
    if (m_prof) m_prof->pop();
    }

void export_MSDAnalyzer()
    {
    class_<MSDAnalyzer, boost::shared_ptr<MSDAnalyzer>, bases<Analyzer>, boost::noncopyable>
    ("MSDAnalyzer", init< boost::shared_ptr<SystemDefinition>, const std::string&, const std::string& >())
    .def("setDelimiter", &MSDAnalyzer::setDelimiter)
    .def("addColumn", &MSDAnalyzer::addColumn)
    ;
    }

