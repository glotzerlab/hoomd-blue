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

#ifndef __GSDDUMPWRITER_H__
#define __GSDDUMPWRITER_H__

#include "Analyzer.h"
#include "ParticleGroup.h"

#include <string>
#include <boost/shared_ptr.hpp>
#include "hoomd/extern/gsd.h"

/*! \file GSDDumpWriter.h
    \brief Declares the GSDDumpWriter class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Analyzer for writing out GSD dump files
/*! GSDDumpWriter writes out the current state of the system to a GSD file
    every time analyze() is called. When a group is specified, only write out the
    particles in the group.

    On the first call to analyze() \a fname is created with a dcd header. If it already
    exists, append to the file (unless the user specifies overwrite=True).

    \ingroup analyzers
*/
class GSDDumpWriter : public Analyzer
    {
    public:
        //! Construct the writer
        GSDDumpWriter(boost::shared_ptr<SystemDefinition> sysdef,
                      const std::string &fname,
                      boost::shared_ptr<ParticleGroup> group,
                      bool overwrite=false,
                      bool truncate=false);

        //! Control attribute writes
        void setWriteAttribute(bool b)
            {
            m_write_attribute = b;
            }

        //! Control property writes
        void setWriteProperty(bool b)
            {
            m_write_property = b;
            }

        //! Control momentum writes
        void setWriteMomentum(bool b)
            {
            m_write_momentum = b;
            }

        //! Control topology writes
        void setWriteTopology(bool b)
            {
            m_write_topology = b;
            }

        //! Destructor
        ~GSDDumpWriter();

        //! Write out the data for the current timestep
        void analyze(unsigned int timestep);

    private:
        std::string m_fname;                //!< The file name we are writing to
        bool m_overwrite;                   //!< True if file should be overwritten
        bool m_truncate;                    //!< True if we should truncate the file on every analyze()
        bool m_is_initialized;              //!< True if the file is open
        bool m_write_attribute;             //!< True if attributes should be written
        bool m_write_property;              //!< True if properties should be written
        bool m_write_momentum;              //!< True if momenta should be written
        bool m_write_topology;              //!< True if topology should be written
        gsd_handle m_handle;                //!< Handle to the file

        boost::shared_ptr<ParticleGroup> m_group;   //!< Group to write out to the file

        //! Write a type mapping out to the file
        void writeTypeMapping(std::string chunk, std::vector< std::string > type_mapping);

        //! Initializes the output file for writing
        void initFileIO();

        //! Write frame header
        void writeFrameHeader(unsigned int timestep);

        //! Write particle attributes
        void writeAttributes(const SnapshotParticleData<float>& snapshot);

        //! Write particle properties
        void writeProperties(const SnapshotParticleData<float>& snapshot);

        //! Write particle momenta
        void writeMomenta(const SnapshotParticleData<float>& snapshot);

        //! Write bond topology
        void writeTopology(BondData::Snapshot& bond,
                           AngleData::Snapshot& angle,
                           DihedralData::Snapshot& dihedral,
                           ImproperData::Snapshot& improper,
                           ConstraintData::Snapshot& constraint);

        //! Check and raise an exception if an error occurs
        void checkError(int retval);
    };

//! Exports the GSDDumpWriter class to python
void export_GSDDumpWriter();

#endif
