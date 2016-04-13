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

/*! \file HOOMDInitializer.h
    \brief Declares the HOOMDInitializer class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "ParticleData.h"
#include <string>
#include "hoomd/extern/gsd.h"

#ifndef __GSD_INITIALIZER_H__
#define __GSD_INITIALIZER_H__

//! Forward declarations
class ExecutionConfiguation;
template <class Real> struct SnapshotSystemData;

//! Reads a GSD input file
/*! Read an input GSD file and generate a system snapshot. GSDReader can read any frame from a GSD
    file into the snapshot. For information on the GSD specification, see http://gsd.readthedocs.org/

    \ingroup data_structs
*/
class GSDReader
    {
    public:
        //! Loads in the file and parses the data
        GSDReader(boost::shared_ptr<const ExecutionConfiguration> exec_conf,
                  const std::string &name,
                  const uint64_t frame);

        //! Destructor
        ~GSDReader();

        //! Returns the timestep of the simulation
        uint64_t getTimeStep() const
            {
            uint64_t timestep = m_timestep;

            // timestep is only read on the root, broadcast to the other nodes
            #ifdef ENABLE_MPI
            const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
            bcast(timestep, 0, mpi_comm);
            #endif

            return timestep;
            }

        //! initializes a snapshot with the particle data
        boost::shared_ptr< SnapshotSystemData<float> > getSnapshot() const
            {
            return m_snapshot;
            }

    private:
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration
        uint64_t m_timestep;                                         //!< Timestep at the selected frame
        std::string m_name;                                          //!< Cached file name
        uint64_t m_frame;                                            //!< Cached frame
        boost::shared_ptr< SnapshotSystemData<float> > m_snapshot;   //!< The snapshot to read
        gsd_handle m_handle;                                         //!< Handle to the file

        //! Helper function to read a quantity from the file
        bool readChunk(void *data, uint64_t frame, const char *name, size_t expected_size, unsigned int cur_n=0);
        //! Helper function to read a type list from the file
        std::vector<std::string> readTypes(uint64_t frame, const char *name);

        // helper functions to read sections of the file
        void readHeader();
        void readParticles();
        void readTopology();
    };

//! Exports GSDReader to python
void export_GSDReader();

#endif
