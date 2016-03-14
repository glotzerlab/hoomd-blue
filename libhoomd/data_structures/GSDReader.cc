/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

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

/*! \file GSDReader.cc
    \brief Defines the GSDReader class
*/

#include "GSDReader.h"
#include "SnapshotSystemData.h"
#include "ExecutionConfiguration.h"
#include <gsd.h>
#include <string.h>

#include <stdexcept>
using namespace std;

#include <boost/python.hpp>

using namespace boost::python;
using namespace boost;

/*! \param exec_conf The execution configuration
    \param name File name to read
    \param frame Frame index to read from the file

    The GSDReader constructor opens the GSD file, initializes an empty snapshot, and reads the file into
    memory (on the root rank).
*/
GSDReader::GSDReader(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
                     const std::string &name,
                     const uint64_t frame)
    : m_exec_conf(exec_conf), m_timestep(0), m_name(name), m_frame(frame)
    {
    #ifdef ENABLE_MPI
    // if we are not the root processor, do not perform file I/O
    if (m_comm && !m_exec_conf->isRoot())
        {
        return;
        }
    #endif

    // open the GSD file in read mode
    m_exec_conf->msg->notice(3) << "data.gsd_snapshot: open gsd file " << name << endl;
    int retval = gsd_open(&m_handle, name.c_str(), GSD_OPEN_READONLY);
    if (retval == -1)
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << strerror(errno) << " - " << name << endl;
        throw runtime_error("Error opening GSD file");
        }
    else if (retval == -2)
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << name << " is not a valid GSD file" << endl;
        throw runtime_error("Error opening GSD file");
        }
    else if (retval == -3)
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Invalid GSD file version in " << name << endl;
        throw runtime_error("Error opening GSD file");
        }
    else if (retval == -4)
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Corrupt GSD file: " << name << endl;
        throw runtime_error("Error opening GSD file");
        }
    else if (retval == -5)
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Out of memory opening: " << name << endl;
        throw runtime_error("Error opening GSD file");
        }
    else if (retval != 0)
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Unknown error opening: " << name << endl;
        throw runtime_error("Error opening GSD file");
        }

    // validate schema
    if (string(m_handle.header.schema) != string("hoomd"))
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Invalid schema in " << name << endl;
        throw runtime_error("Error opening GSD file");
        }
    if (m_handle.header.schema_version != gsd_make_version(0,1))
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Invalid schema version in " << name << endl;
        throw runtime_error("Error opening GSD file");
        }

    // validate number of frames
    if (frame >= gsd_get_nframes(&m_handle))
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Cannot read frame " << frame << " " << name << " only has " << gsd_get_nframes(&m_handle) << " frames" << endl;
        throw runtime_error("Error opening GSD file");
        }

    m_snapshot = boost::shared_ptr< SnapshotSystemData<float> >(new SnapshotSystemData<float>);

    readHeader();
    readAttributes();
    readProperties();
    readMomenta();
    readTopology();
    }

GSDReader::~GSDReader()
    {
    gsd_close(&m_handle);
    }

/*! \param data Pointer to data to read into
    \param frame Frame index to read from
    \param name Name of the data chunk
    \param exepected_size Expected size of the data chunk in bytes.

    Attempts to read the data chunk of the given name at the given frame. If it is not present at this
    frame, attempt to read from frame 0. If it is also not present at frame 0, return false.
    If the found data chunk is not the expected size, throw an exception.

    Return true if data is actually read from the file.
*/
bool GSDReader::readChunk(void *data, uint64_t frame, const char *name, size_t expected_size)
    {
    const struct gsd_index_entry* entry = gsd_find_chunk(&m_handle, frame, name);
    if (entry == NULL && frame != 0)
        entry = gsd_find_chunk(&m_handle, 0, name);

    if (entry == NULL)
        return false;
    else
        {
        size_t actual_size = entry->N * entry->M * gsd_sizeof_type((enum gsd_type)entry->type);
        if (actual_size != expected_size)
            {
            m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Expecting " << expected_size << " bytes in" << name << " but found " << actual_size << endl;
            throw runtime_error("Error reading GSD file");
            }
        int retval = gsd_read_chunk(&m_handle, data, entry);

        if (retval == -1)
            {
            m_exec_conf->msg->error() << "data.gsd_snapshot: " << strerror(errno) << " - " << m_name << endl;
            throw runtime_error("Error reading GSD file");
            }
        else if (retval == -2)
            {
            m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Unknown error reading: " << m_name << endl;
            throw runtime_error("Error reading GSD file");
            }
        else if (retval == -3)
            {
            m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Invalid GSD file " << m_name << endl;
            throw runtime_error("Error reading GSD file");
            }
        else if (retval != 0)
            {
            m_exec_conf->msg->error() << "data.gsd_snapshot: " << "Unknown error reading: " << m_name << endl;
            throw runtime_error("Error reading GSD file");
            }

        return true;
        }
    }

/*! Read the same data chunks written by GSDDumpWriter::writeFrameHeader
*/
void GSDReader::readHeader()
    {
    readChunk(&m_timestep, m_frame, "configuration/step", 8);

    uint8_t dim = 3;
    readChunk(&dim, m_frame, "configuration/dimensions", 1);
    m_snapshot->dimensions = dim;

    float box[6] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    readChunk(&box, m_frame, "configuration/box", 6*4);
    m_snapshot->global_box = BoxDim(box[0], box[1], box[2]);
    m_snapshot->global_box.setTiltFactors(box[3], box[4], box[5]);

    unsigned int N = 0;
    readChunk(&N, m_frame, "particles/N", 4);
    if (N == 0)
        {
        m_exec_conf->msg->error() << "data.gsd_snapshot: " << "cannot read a file with 0 particles" << endl;
        throw runtime_error("Error reading GSD file");
        }
    m_snapshot->particle_data.resize(N);
    }

/*! Read the same data chunks written by GSDDumpWriter::writeAttributes
*/
void GSDReader::readAttributes()
    {
    }

/*! Read the same data chunks written by GSDDumpWriter::writeProperties
*/
void GSDReader::readProperties()
    {
    }

/*! Read the same data chunks written by GSDDumpWriter::writeMomenta
*/
void GSDReader::readMomenta()
    {
    }

/*! Read the same data chunks written by GSDDumpWriter::writeTopology
*/
void GSDReader::readTopology()
    {
    // TODO
    }

void export_GSDReader()
    {
    class_< GSDReader >("GSDReader", init<boost::shared_ptr<const ExecutionConfiguration>, const string&, const uint64_t>())
    .def("getTimeStep", &GSDReader::getTimeStep)
    .def("getSnapshot", &GSDReader::getSnapshot)
    ;
    }
