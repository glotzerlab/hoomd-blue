/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#ifndef __DCDDUMPWRITER_H__
#define __DCDDUMPWRITER_H__

#include <string>
#include <boost/shared_ptr.hpp>
#include <fstream>
#include "Analyzer.h"
#include "ParticleGroup.h"

/*! \file DCDDumpWriter.h
    \brief Declares the DCDDumpWriter class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

// The DCD Dump writer is based on code from the molfile plugin to VMD
// and is use under the following license

// University of Illinois Open Source License
// Copyright 2003 Theoretical and Computational Biophysics Group,
// All rights reserved.

// Developed by:       Theoretical and Computational Biophysics Group
//             University of Illinois at Urbana-Champaign
//            http://www.ks.uiuc.edu/

//! Analyzer for writing out DCD dump files
/*! DCDDumpWriter writes out the current position of all particles to a DCD file
    every time analyze() is called. Use it to create a DCD trajectory for loading
    into VMD.

    On the first call to analyze() \a fname is created with a dcd header. If the file already exists,
    it is overwritten.

    Due to a limitation in the DCD format, the time step period between calls to
    analyze() \b must be specified up front. If analyze() detects that this period is
    not being maintained, it will print a warning but continue.
    \ingroup analyzers
*/
class DCDDumpWriter : public Analyzer
    {
    public:
        //! Construct the writer
        DCDDumpWriter(boost::shared_ptr<SystemDefinition> sysdef,
                      const std::string &fname,
                      unsigned int period,
                      boost::shared_ptr<ParticleGroup> group,
                      bool overwrite=false);

        //! Destructor
        ~DCDDumpWriter();

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

        //! Set whether coordinates should be written out wrapped or unwrapped.
        void setUnwrapFull(bool enable)
            {
            m_unwrap_full = enable;
            }

        //! Set whether rigid body coordinates should be written out wrapped or unwrapped.
        void setUnwrapRigid(bool enable)
            {
            m_unwrap_rigid = enable;
            }

        //! Set whether the z-component should be overwritten by the orientation angle
        void setAngleZ(bool enable)
            {
            m_angle = enable;
            }

    private:
        std::string m_fname;                //!< The file name we are writing to
        unsigned int m_start_timestep;      //!< First time step written to the file
        unsigned int m_period;              //!< Time step period bewteen writes
        boost::shared_ptr<ParticleGroup> m_group; //!< Group of particles to write to the DCD file
        boost::shared_ptr<RigidData> m_rigid_data; //!< For accessing rigid body data
        unsigned int m_num_frames_written;  //!< Count the number of frames written to the file
        unsigned int m_last_written_step;   //!< Last timestep written in a a file we are appending to
        bool m_appending;                   //!< True if this instance is appending to an existing DCD file
        bool m_unwrap_full;                 //!< True if coordinates should be written out fully unwrapped in the box
        bool m_unwrap_rigid;                //!< True if rigid bodies should be written out unwrapped
        bool m_angle;                       //!< True if the z-component should be set to the orientation angle

        bool m_overwrite;                   //!< True if file should be overwritten
        bool m_is_initialized;              //!< True if file IO has been initialized

        float *m_staging_buffer;            //!< Buffer for staging particle positions in tag order

        // helper functions

        //! Initalizes the file header
        void write_file_header(std::fstream &file);
        //! Writes the frame header
        void write_frame_header(std::fstream &file);
        //! Writes the particle positions for a frame
        void write_frame_data(std::fstream &file, const SnapshotParticleData& snapshot);
        //! Updates the file header
        void write_updated_header(std::fstream &file, unsigned int timestep);
        //! Initializes the output file for writing
        void initFileIO();

    };

//! Exports the DCDDumpWriter class to python
void export_DCDDumpWriter();

#endif
