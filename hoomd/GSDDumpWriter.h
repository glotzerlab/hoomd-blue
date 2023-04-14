// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "Analyzer.h"
#include "ParticleGroup.h"
#include "SharedSignal.h"

#include "hoomd/extern/gsd.h"
#include <memory>
#include <string>

/*! \file GSDDumpWriter.h
    \brief Declares the GSDDumpWriter class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
//! Analyzer for writing out GSD dump files
/*! GSDDumpWriter writes out the current state of the system to a GSD file
    every time analyze() is called. When a group is specified, only write out the
    particles in the group.

    The file is not opened until the first call to analyze().

    \ingroup analyzers
*/
class PYBIND11_EXPORT GSDDumpWriter : public Analyzer
    {
    public:
    //! Construct the writer
    GSDDumpWriter(std::shared_ptr<SystemDefinition> sysdef,
                  std::shared_ptr<Trigger> trigger,
                  const std::string& fname,
                  std::shared_ptr<ParticleGroup> group,
                  std::string mode = "ab",
                  bool truncate = false);

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

    std::string getFilename()
        {
        return m_fname;
        }

    std::string getMode()
        {
        return m_mode;
        }

    bool getTruncate()
        {
        return m_truncate;
        }

    std::shared_ptr<ParticleGroup> getGroup()
        {
        return m_group;
        }

    pybind11::tuple getDynamic()
        {
        pybind11::list result;
        if (m_write_attribute)
            result.append("attribute");
        if (m_write_property)
            result.append("property");
        if (m_write_momentum)
            result.append("momentum");
        if (m_write_topology)
            result.append("topology");

        return pybind11::tuple(result);
        }

    //! Destructor
    ~GSDDumpWriter();

    //! Write out the data for the current timestep
    void analyze(uint64_t timestep);

    hoomd::detail::SharedSignal<int(gsd_handle&)>& getWriteSignal()
        {
        return m_write_signal;
        }

    /// Write a logged quantities
    void writeLogQuantities(pybind11::dict dict);

    /// Set the log writer
    void setLogWriter(pybind11::object log_writer)
        {
        m_log_writer = log_writer;
        }

    /// Get the log writer
    pybind11::object getLogWriter()
        {
        return m_log_writer;
        }

    /// Get needed pdata flags
    virtual PDataFlags getRequestedPDataFlags()
        {
        PDataFlags flags;

        if (!m_log_writer.is_none())
            {
            flags.set();
            }

        return flags;
        }

    /// Get the write_diameter flag
    bool getWriteDiameter()
        {
        return m_write_diameter;
        }

    /// Set the write_diameter flag
    void setWriteDiameter(bool write_diameter)
        {
        m_write_diameter = write_diameter;
        }

    protected:

    /// Store a GSD frame for writing.
    /** Local frames store particles local to the rank, sorted in ascending tag order.
        Global frames store the entire system, sorted in ascending tag order.

        Entries with 0 sized vectors should not be written to the file.

        Note: In the first implementation, only particle data is local/global .
        The bond/angle/dihedral/etc... data stored in the *local* frame is actually global.
    */
    struct GSDFrame
        {
        uint64_t timestep;
        BoxDim global_box;
        SnapshotParticleData<float> particle_data;
        BondData::Snapshot bond_data;
        AngleData::Snapshot angle_data;
        DihedralData::Snapshot dihedral_data;
        ImproperData::Snapshot improper_data;
        ConstraintData::Snapshot constraint_data;
        PairData::Snapshot pair_data;
        };

    /// Flags for all_default bitset
    struct gsd_flag
        {
        enum Enum
            {
            orientation,
            type,
            mass,
            charge,
            diameter,
            body,
            inertia,
            velocity,
            angmom,
            image,
            };
        };

    private:
    std::string m_fname;            //!< The file name we are writing to
    std::string m_mode;             //!< The file open mode
    bool m_truncate = false;        //!< True if we should truncate the file on every analyze()
    bool m_is_initialized = false;  //!< True if the file is open
    bool m_write_attribute = false; //!< True if attributes should be written
    bool m_write_property = false;  //!< True if properties should be written
    bool m_write_momentum = false;  //!< True if momenta should be written
    bool m_write_topology = false;  //!< True if topology should be written
    bool m_write_diameter = false;  //!< True if the diameter attribute should be written
    gsd_handle m_handle;            //!< Handle to the file

    /// Number of frames written to the file.
    uint64_t m_nframes = 0;

    static std::list<std::string> particle_chunks;

    /// Callback to write log quantities to file
    pybind11::object m_log_writer;

    std::shared_ptr<ParticleGroup> m_group; //!< Group to write out to the file
    std::unordered_map<std::string, bool>
        m_nondefault; //!< Map of quantities (true when non-default in frame 0)

    hoomd::detail::SharedSignal<int(gsd_handle&)> m_write_signal;

    GSDFrame m_local_frame;
    GSDFrame m_global_frame;

    //! Write a type mapping out to the file
    void writeTypeMapping(std::string chunk, std::vector<std::string> type_mapping);

    //! Initializes the output file for writing
    void initFileIO();

    //! Write frame header
    void writeFrameHeader(const GSDFrame& frame);

    //! Write particle attributes
    void writeAttributes(const GSDFrame& frame);

    //! Write particle properties
    void writeProperties(const GSDFrame& frame);

    //! Write particle momenta
    void writeMomenta(const GSDFrame& frame);

    //! Write bond topology
    void writeTopology(BondData::Snapshot& bond,
                       AngleData::Snapshot& angle,
                       DihedralData::Snapshot& dihedral,
                       ImproperData::Snapshot& improper,
                       ConstraintData::Snapshot& constraint,
                       PairData::Snapshot& pair);

    //! Check and raise an exception if an error occurs
    void checkError(int retval);

    //! Populate the non-default map
    void populateNonDefault();

    /// Populate local frame with data.
    void populateLocalFrame(GSDFrame& frame, uint64_t timestep);

    friend void export_GSDDumpWriter(pybind11::module& m);
    };

namespace detail
    {
//! Exports the GSDDumpWriter class to python
void export_GSDDumpWriter(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
