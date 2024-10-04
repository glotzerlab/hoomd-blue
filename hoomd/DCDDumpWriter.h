// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __DCDDUMPWRITER_H__
#define __DCDDUMPWRITER_H__

#include "Analyzer.h"
#include "ParticleGroup.h"

#include <fstream>
#include <memory>
#include <string>

/*! \file DCDDumpWriter.h
    \brief Declares the DCDDumpWriter class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

// The DCD Dump writer is based on code from the molfile plugin to VMD
// and is use under the following license

// University of Illinois Open Source License
// Copyright 2003 Theoretical and Computational Biophysics Group,
// All rights reserved.

// Developed by:       Theoretical and Computational Biophysics Group
//             University of Illinois at Urbana-Champaign
//            https://www.ks.uiuc.edu/

namespace hoomd
    {
//! Analyzer for writing out DCD dump files
/*! DCDDumpWriter writes out the current position of all particles to a DCD file
    every time analyze() is called. Use it to create a DCD trajectory for loading
    into VMD.

    On the first call to analyze() \a fname is created with a dcd header. If the file already
   exists, it is overwritten.

    Due to a limitation in the DCD format, the time step period between calls to
    analyze() \b must be specified up front. If analyze() detects that this period is
    not being maintained, it will print a warning but continue.
    \ingroup analyzers
*/
class PYBIND11_EXPORT DCDDumpWriter : public Analyzer
    {
    public:
    //! Construct the writer
    DCDDumpWriter(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<Trigger> trigger,
                  const std::string& fname,
                  unsigned int period,
                  std::shared_ptr<ParticleGroup> group,
                  bool overwrite = false);

    //! Destructor
    ~DCDDumpWriter();

    //! Write out the data for the current timestep
    void analyze(uint64_t timestep);

    //! Set whether coordinates should be written out wrapped or unwrapped.
    void setUnwrapFull(bool enable)
        {
        m_unwrap_full = enable;
        }

    bool getUnwrapFull()
        {
        return m_unwrap_full;
        }

    //! Set whether rigid body coordinates should be written out wrapped or unwrapped.
    void setUnwrapRigid(bool enable)
        {
        m_unwrap_rigid = enable;
        }

    bool getUnwrapRigid()
        {
        return m_unwrap_rigid;
        }

    //! Set whether the z-component should be overwritten by the orientation angle
    void setAngleZ(bool enable)
        {
        m_angle = enable;
        }

    bool getAngleZ()
        {
        return m_angle;
        }

    bool getOverwrite()
        {
        return m_overwrite;
        }

    private:
    std::string m_fname;                    //!< The file name we are writing to
    uint64_t m_start_timestep;              //!< First time step written to the file
    unsigned int m_period;                  //!< Time step period between writes
    std::shared_ptr<ParticleGroup> m_group; //!< Group of particles to write to the DCD file
    unsigned int m_num_frames_written;      //!< Count the number of frames written to the file
    unsigned int m_last_written_step; //!< Last timestep written in a a file we are appending to
    bool m_appending;    //!< True if this instance is appending to an existing DCD file
    bool m_unwrap_full;  //!< True if coordinates should be written out fully unwrapped in the box
    bool m_unwrap_rigid; //!< True if rigid bodies should be written out unwrapped
    bool m_angle;        //!< True if the z-component should be set to the orientation angle

    bool m_overwrite;       //!< True if file should be overwritten
    bool m_is_initialized;  //!< True if file IO has been initialized
    unsigned int m_nglobal; //!< Initial number of particles

    float* m_staging_buffer; //!< Buffer for staging particle positions in tag order
    std::fstream m_file;     //!< The file object

    // helper functions

    //! Initializes the file header
    void write_file_header(std::fstream& file);
    //! Writes the frame header
    void write_frame_header(std::fstream& file);
    //! Writes the particle positions for a frame
    void write_frame_data(std::fstream& file, const SnapshotParticleData<Scalar>& snapshot);
    //! Updates the file header
    void write_updated_header(std::fstream& file, uint64_t timestep);
    //! Initializes the output file for writing
    void initFileIO(uint64_t timestep);
    };

namespace detail
    {
//! Exports the DCDDumpWriter class to python
void export_DCDDumpWriter(pybind11::module& m);
    } // end namespace detail

    } // end namespace hoomd
#endif
