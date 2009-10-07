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

/*! \file DCDDumpWriter.cc
    \brief Defines the DCDDumpWriter class and related helper functions
*/

// disable microsoft's security deprecation warnings
#ifdef WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <stdexcept>

#include "DCDDumpWriter.h"
#include "time.h"

#include <boost/python.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/convenience.hpp>
using namespace boost::filesystem;
using namespace boost::python;
using namespace std;

//! File position of NFILE in DCD header
#define NFILE_POS 8L
//! File position of NSTEP in DCD header
#define NSTEP_POS 20L

//! simple helper function to write an integer
/*! \param file file to write to
    \param val integer to write
*/
static void write_int(fstream &file, unsigned int val)
    {
    file.write((char *)&val, sizeof(unsigned int));
    }

//! simple helper function to read in integer
/*! \param file file to read from
    \returns integer read
*/
static unsigned int read_int(fstream &file)
    {
    unsigned int val;
    file.read((char *)&val, sizeof(unsigned int));
    return val;
    }

/*! Constructs the DCDDumpWriter. After construction, settings are set. No file operations are
    attempted until analyze() is called.

    \param sysdef SystemDefinition containing the ParticleData to dump
    \param fname File name to write DCD data to
    \param period Period which analyze() is going to be called at
    \param overwrite If false, existing files will be appended to. If true, existing files will be overwritten.

    \note You must call analyze() with the same period specified in the constructor or
    the time step inforamtion in the file will be invalid. analyze() will print a warning
    if it is called out of sequence.
*/
DCDDumpWriter::DCDDumpWriter(boost::shared_ptr<SystemDefinition> sysdef, const std::string &fname, unsigned int period, bool overwrite) : Analyzer(sysdef), m_fname(fname), m_start_timestep(0), m_period(period), m_num_frames_written(0), m_last_written_step(0), m_appending(false)
    {
    // handle appending to an existing file if it is requested
    if (!overwrite && exists(fname))
        {
        cout << "Notice: Appending to existing DCD file \"" << fname << "\"" << endl;
        
        // open the file and get data from the header
        fstream file;
        file.open(m_fname.c_str(), ios::ate | ios::in | ios::out | ios::binary);
        file.seekp(NFILE_POS);
        
        m_num_frames_written = read_int(file);
        m_start_timestep = read_int(file);
        unsigned int file_period = read_int(file);
        
        // warn the user if we are now dumping at a different period
        if (file_period != m_period)
            cout << "***Warning! DCDDumpWriter is appending to a file that has period " << file_period << " that is not the same as the requested period of " << m_period << endl;
            
        m_last_written_step = read_int(file);
        
        // check for errors
        if (!file.good())
            {
            cerr << endl << "***Error! Error reading DCD header data" << endl << endl;
            throw runtime_error("Error appending to DCD file");
            }
            
        m_appending = true;
        }
        
    m_staging_buffer = new Scalar[m_pdata->getN()];
    }

DCDDumpWriter::~DCDDumpWriter()
    {
    delete[] m_staging_buffer;
    }

/*! \param timestep Current time step of the simulation
    The very first call to analyze() will result in the creation (or overwriting) of the
    file fname and the writing of the current timestep snapshot. After that, each call to analyze
    will add a new snapshot to the end of the file.
*/
void DCDDumpWriter::analyze(unsigned int timestep)
    {
    // the file object
    fstream file;
    
    // initialize the file on the first frame written
    if (m_num_frames_written == 0)
        {
        // open the file and truncate it
        file.open(m_fname.c_str(), ios::trunc | ios::out | ios::binary);
        
        // write the file header
        m_start_timestep = timestep;
        write_file_header(file);
        }
    else
        {
        if (m_appending && timestep <= m_last_written_step)
            {
            cout << "***Warning! DCDDumpWriter is not writing output at timestep " << timestep << " because the file reports that it already has data up to step " << m_last_written_step << endl;
            return;
            }
            
        // open the file and move the file pointer to the end
        file.open(m_fname.c_str(), ios::ate | ios::in | ios::out | ios::binary);
        
        // verify the period on subsequent frames
        if ( (timestep - m_start_timestep) % m_period != 0)
            cout << "***Warning! DCDDumpWriter is writing time step " << timestep << " which is not specified in the period of the DCD file: " << m_start_timestep << " + i * " << m_period << endl;
        }
        
    // write the data for the current time step
    write_frame_header(file);
    write_frame_data(file);
    
    // update the header with the number of frames written
    m_num_frames_written++;
    write_updated_header(file, timestep);
    file.close();
    }

/*! \param file File to write to
    Writes the initial DCD header to the beginning of the file. This must be
    called on a newly created (or truncated file).
*/
void DCDDumpWriter::write_file_header(std::fstream &file)
    {
    // the first 4 bytes in the file must be 84
    write_int(file, 84);
    
    // the next 4 bytes in the file must be "CORD"
    char cord_data[] = "CORD";
    file.write(cord_data, 4);
    write_int(file, 0);      // Number of frames in file, none written yet
    write_int(file, m_start_timestep); // Starting timestep
    write_int(file, m_period);  // Timesteps between frames written to the file
    write_int(file, 0);      // Number of timesteps in simulation
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);         // timestep (unused)
    write_int(file, 1);         // include unit cell
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 0);
    write_int(file, 24);    // Pretend to be CHARMM version 24
    write_int(file, 84);
    write_int(file, 164);
    write_int(file, 2);
    
    char title_string[81];
    memset(title_string, 0, 81);
    char remarks[] = "Created by HOOMD";
    strncpy(title_string, remarks, 80);
    title_string[79] = '\0';
    file.write(title_string, 80);
    
    char time_str[81];
    memset(time_str, 0, 81);
    time_t cur_time = time(NULL);
    tm *tmbuf=localtime(&cur_time);
    strftime(time_str, 80, "REMARKS Created  %d %B, %Y at %H:%M", tmbuf);
    file.write(time_str, 80);
    
    write_int(file, 164);
    write_int(file, 4);
    write_int(file, m_pdata->getN());
    write_int(file, 4);
    
    // check for errors
    if (!file.good())
        {
        cerr << endl << "***Error! Error writing DCD header" << endl << endl;
        throw runtime_error("Error writing DCD file");
        }
    }

/*! \param file File to write to
    Writes the header that precedes each snapshot in the file. This header
    includes information on the box size of the simulation.
*/
void DCDDumpWriter::write_frame_header(std::fstream &file)
    {
    double unitcell[6];
    BoxDim box = m_pdata->getBox();
    // set box dimensions
    unitcell[0] = box.xhi - box.xlo;
    unitcell[2] = box.yhi - box.ylo;
    unitcell[5] = box.zhi - box.zlo;
    // box angles are 90 degrees
    unitcell[1] = 0.0f;
    unitcell[3] = 0.0f;
    unitcell[4] = 0.0f;
    
    write_int(file, 48);
    file.write((char *)unitcell, 48);
    write_int(file, 48);
    
    // check for errors
    if (!file.good())
        {
        cerr << endl << "***Error! Error writing DCD frame header" << endl << endl;
        throw runtime_error("Error writing DCD file");
        }
    }

/*! \param file File to write to
    Writes the actual particle positions for all particles at the current time step
*/
void DCDDumpWriter::write_frame_data(std::fstream &file)
    {
    // we need to unsort the positions and write in tag order
    assert(m_staging_buffer);
    
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    
    // prepare x coords for writing
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        m_staging_buffer[i] = arrays.x[arrays.rtag[i]];
    // write x coords
    write_int(file, m_pdata->getN() * sizeof(float));
    file.write((char *)m_staging_buffer, m_pdata->getN() * sizeof(float));
    write_int(file, m_pdata->getN() * sizeof(float));
    
    // prepare y coords for writing
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        m_staging_buffer[i] = arrays.y[arrays.rtag[i]];
    // write y coords
    write_int(file, m_pdata->getN() * sizeof(float));
    file.write((char *)m_staging_buffer, m_pdata->getN() * sizeof(float));
    write_int(file, m_pdata->getN() * sizeof(float));
    
    // prepare z coords for writing
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        m_staging_buffer[i] = arrays.z[arrays.rtag[i]];
    // write z coords
    write_int(file, m_pdata->getN() * sizeof(float));
    file.write((char *)m_staging_buffer, m_pdata->getN() * sizeof(float));
    write_int(file, m_pdata->getN() * sizeof(float));
    
    m_pdata->release();
    
    // check for errors
    if (!file.good())
        {
        cerr << endl << "***Error! Error writing DCD frame data" << endl << endl;
        throw runtime_error("Error writing DCD file");
        }
    }

/*! \param file File to write to
    \param timestep Current time step of the simulation

    Updates the pointers in the main file header to reflect the current number of frames
    written and the last time step written.
*/
void DCDDumpWriter::write_updated_header(std::fstream &file, unsigned int timestep)
    {
    file.seekp(NFILE_POS);
    write_int(file, m_num_frames_written);
    
    file.seekp(NSTEP_POS);
    write_int(file, timestep);
    }

void export_DCDDumpWriter()
    {
    class_<DCDDumpWriter, boost::shared_ptr<DCDDumpWriter>, bases<Analyzer>, boost::noncopyable>
    ("DCDDumpWriter", init< boost::shared_ptr<SystemDefinition>, std::string, unsigned int, bool>())
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

