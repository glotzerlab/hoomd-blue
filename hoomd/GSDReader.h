// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file HOOMDInitializer.h
    \brief Declares the HOOMDInitializer class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ParticleData.h"
#include <string>
#include "hoomd/extern/gsd.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifndef __GSD_INITIALIZER_H__
#define __GSD_INITIALIZER_H__

//! Forward declarations
template <class Real> struct SnapshotSystemData;

//! Reads a GSD input file
/*! Read an input GSD file and generate a system snapshot. GSDReader can read any frame from a GSD
    file into the snapshot. For information on the GSD specification, see http://gsd.readthedocs.io/

    \ingroup data_structs
*/
class PYBIND11_EXPORT GSDReader
    {
    public:
        //! Loads in the file and parses the data
        GSDReader(std::shared_ptr<const ExecutionConfiguration> exec_conf,
                  const std::string &name,
                  const uint64_t frame,
                  bool from_end);

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
        std::shared_ptr< SnapshotSystemData<float> > getSnapshot() const
            {
            return m_snapshot;
            }

        //! initializes a snapshot with the particle data
        uint64_t getFrame() const
            {
            return m_frame;
            }

        //! Helper function to read a quantity from the file
        bool readChunk(void *data, uint64_t frame, const char *name, size_t expected_size, unsigned int cur_n=0);

        //! clears the snapshot object
        void clearSnapshot()
            {
            m_snapshot.reset();
            }

        //! get handle
        gsd_handle getHandle(void) const
            {
            return m_handle;
            }

        pybind11::list readTypeShapesPy(uint64_t frame);

    private:
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< The execution configuration
        uint64_t m_timestep;                                         //!< Timestep at the selected frame
        std::string m_name;                                          //!< Cached file name
        uint64_t m_frame;                                            //!< Cached frame
        std::shared_ptr< SnapshotSystemData<float> > m_snapshot;   //!< The snapshot to read
        gsd_handle m_handle;                                         //!< Handle to the file

        //! Helper function to read a type list from the file
        std::vector<std::string> readTypes(uint64_t frame, const char *name);

        // helper functions to read sections of the file
        void readHeader();
        void readParticles();
        void readTopology();

        /// Check and raise an exception if an error occurs
        void checkError(int retval);
    };

/** Read state information from a GSD file

    GSDStateReader provides an interface for ``from_state`` methods to discover and read state data
    from a GSD file.
*/
class PYBIND11_EXPORT GSDStateReader
    {
    public:
        /** Open the file

            @param name File name to open.
            @frame Index of frame to access. Negative values index from the end.
        */
        GSDStateReader(const std::string &name, const int64_t frame);

        /// Destructor
        ~GSDStateReader();

        /// Get a list of chunk names starting with *base*.
        std::vector<std::string> getAvailableChunks(const std::string& base);

        /// Read a chunk and return as a numpy array
        pybind11::array readChunk(const std::string& name);

    private:
        /// Store the filename
        std::string m_name;

        /// Frame to read from the file
        uint64_t m_frame;

        /// Handle to the file
        gsd_handle m_handle;

        /// Check and raise an exception if an error occurs
        void checkError(int retval);
    };


/// Exports GSDReader and GSDStateReader to python
void export_GSDReader(pybind11::module& m);

#endif
