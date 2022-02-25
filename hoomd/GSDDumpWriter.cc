// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "GSDDumpWriter.h"
#include "Filesystem.h"
#include "GSD.h"
#include "HOOMDVersion.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include <limits>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string.h>
using namespace std;
using namespace hoomd::detail;

namespace hoomd
    {
std::list<std::string> GSDDumpWriter::particle_chunks {"particles/typeid",
                                                       "particles/mass",
                                                       "particles/charge",
                                                       "particles/diameter",
                                                       "particles/body",
                                                       "particles/moment_inertia",
                                                       "particles/orientation",
                                                       "particles/velocity",
                                                       "particles/angmom",
                                                       "particles/image"};

/*! Constructs the GSDDumpWriter. After construction, settings are set. No file operations are
    attempted until analyze() is called.

    \param sysdef SystemDefinition containing the ParticleData to dump
    \param fname File name to write data to
    \param group Group of particles to include in the output
    \param mode File open mode ("wb", "xb", or "ab")
    \param truncate If true, truncate the file to 0 frames every time analyze() called, then write
   out one frame

    If the group does not include all particles, then topology information cannot be written to the
   file.
*/
GSDDumpWriter::GSDDumpWriter(std::shared_ptr<SystemDefinition> sysdef,
                             const std::string& fname,
                             std::shared_ptr<ParticleGroup> group,
                             std::string mode,
                             bool truncate)
    : Analyzer(sysdef), m_fname(fname), m_mode(mode), m_truncate(truncate), m_is_initialized(false),
      m_group(group)
    {
    m_exec_conf->msg->notice(5) << "Constructing GSDDumpWriter: " << m_fname << " " << mode << " "
                                << truncate << endl;
    if (mode != "wb" && mode != "xb" && mode != "ab")
        {
        throw std::invalid_argument("Invalid GSD file mode: " + mode);
        }
    m_log_writer = pybind11::none();
    }

//! Initializes the output file for writing
void GSDDumpWriter::initFileIO()
    {
    // create a new file or overwrite an existing one
    if (m_mode == "wb" || m_mode == "xb" || (m_mode == "ab" && !filesystem::exists(m_fname)))
        {
        ostringstream o;
        o << "HOOMD-blue " << HOOMD_VERSION;

        m_exec_conf->msg->notice(3) << "GSD: create or overwrite gsd file " << m_fname << endl;
        int retval = gsd_create_and_open(&m_handle,
                                         m_fname.c_str(),
                                         o.str().c_str(),
                                         "hoomd",
                                         gsd_make_version(1, 4),
                                         GSD_OPEN_APPEND,
                                         m_mode == "xb");
        GSDUtils::checkError(retval, m_fname);

        // in a created or overwritten file, all quantities are default
        for (auto const& chunk : particle_chunks)
            {
            m_nondefault[chunk] = false;
            }
        }
    else if (m_mode == "ab")
        {
        // populate the non-default map
        populateNonDefault();

        // open the file in append mode
        m_exec_conf->msg->notice(3) << "GSD: open gsd file " << m_fname << endl;
        int retval = gsd_open(&m_handle, m_fname.c_str(), GSD_OPEN_APPEND);
        GSDUtils::checkError(retval, m_fname);

        // validate schema
        if (string(m_handle.header.schema) != string("hoomd"))
            {
            std::ostringstream s;
            s << "GSD: "
              << "Invalid schema in " << m_fname;
            throw runtime_error("Error opening GSD file");
            }
        if (m_handle.header.schema_version >= gsd_make_version(2, 0))
            {
            std::ostringstream s;
            s << "GSD: "
              << "Invalid schema version in " << m_fname;
            throw runtime_error("Error opening GSD file");
            }
        }
    else
        {
        throw std::invalid_argument("Invalid GSD file mode: " + m_mode);
        }

    m_is_initialized = true;
    }

GSDDumpWriter::~GSDDumpWriter()
    {
    m_exec_conf->msg->notice(5) << "Destroying GSDDumpWriter" << endl;

    bool root = true;
#ifdef ENABLE_MPI
    root = m_exec_conf->isRoot();
#endif

    if (root && m_is_initialized)
        {
        m_exec_conf->msg->notice(5) << "GSD: close gsd file " << m_fname << endl;
        gsd_close(&m_handle);
        }
    }

/*! \param timestep Current time step of the simulation

    The first call to analyze() will create or overwrite the file and write out the current system
   configuration as frame 0. Subsequent calls will append frames to the file, or keep overwriting
   frame 0 if m_truncate is true.
*/
void GSDDumpWriter::analyze(uint64_t timestep)
    {
    Analyzer::analyze(timestep);
    int retval;
    bool root = true;

    // take particle data snapshot
    m_exec_conf->msg->notice(10) << "GSD: taking particle data snapshot" << endl;
    SnapshotParticleData<float> snapshot;
    const std::map<unsigned int, unsigned int>& map = m_pdata->takeSnapshot<float>(snapshot);

#ifdef ENABLE_MPI
    // if we are not the root processor, do not perform file I/O
    root = m_exec_conf->isRoot();
#endif

    // open the file if it is not yet opened
    if (!m_is_initialized && root)
        initFileIO();

    // truncate the file if requested
    if (m_truncate && root)
        {
        m_exec_conf->msg->notice(10) << "GSD: truncating file" << endl;
        retval = gsd_truncate(&m_handle);
        GSDUtils::checkError(retval, m_fname);
        }

    uint64_t nframes = 0;
    if (root)
        {
        nframes = gsd_get_nframes(&m_handle);
        m_exec_conf->msg->notice(10)
            << "GSD: " << m_fname << " has " << nframes << " frames" << endl;
        }

#ifdef ENABLE_MPI
    bcast(nframes, 0, m_exec_conf->getMPICommunicator());
#endif

    if (root)
        {
        // write out the frame header on all frames
        writeFrameHeader(timestep);

        // only write out data chunk categories if requested, or if on frame 0
        if (m_write_attribute || nframes == 0)
            writeAttributes(snapshot, map);
        if (m_write_property || nframes == 0)
            writeProperties(snapshot, map);
        if (m_write_momentum || nframes == 0)
            writeMomenta(snapshot, map);
        }

    // topology is only meaningful if this is the all group
    if (m_group->getNumMembersGlobal() == m_pdata->getNGlobal()
        && (m_write_topology || nframes == 0))
        {
        BondData::Snapshot bdata_snapshot;
        m_sysdef->getBondData()->takeSnapshot(bdata_snapshot);

        AngleData::Snapshot adata_snapshot;
        m_sysdef->getAngleData()->takeSnapshot(adata_snapshot);

        DihedralData::Snapshot ddata_snapshot;
        m_sysdef->getDihedralData()->takeSnapshot(ddata_snapshot);

        ImproperData::Snapshot idata_snapshot;
        m_sysdef->getImproperData()->takeSnapshot(idata_snapshot);

        ConstraintData::Snapshot cdata_snapshot;
        m_sysdef->getConstraintData()->takeSnapshot(cdata_snapshot);

        PairData::Snapshot pdata_snapshot;
        m_sysdef->getPairData()->takeSnapshot(pdata_snapshot);

        if (root)
            writeTopology(bdata_snapshot,
                          adata_snapshot,
                          ddata_snapshot,
                          idata_snapshot,
                          cdata_snapshot,
                          pdata_snapshot);
        }

    // emit on all ranks, the slot needs to handle the mpi logic.
    m_write_signal.emit(m_handle);

    if (!m_log_writer.is_none())
        {
        m_log_writer.attr("_write_frame")(this);
        }

    if (root)
        {
        m_exec_conf->msg->notice(10) << "GSD: ending frame" << endl;
        retval = gsd_end_frame(&m_handle);
        GSDUtils::checkError(retval, m_fname);
        }
    }

void GSDDumpWriter::writeTypeMapping(std::string chunk, std::vector<std::string> type_mapping)
    {
    int max_len = 0;
    for (unsigned int i = 0; i < type_mapping.size(); i++)
        {
        max_len = std::max(max_len, (int)type_mapping[i].size());
        }
    max_len += 1; // for null

        {
        m_exec_conf->msg->notice(10) << "GSD: writing " << chunk << endl;
        std::vector<char> types(max_len * type_mapping.size());
        for (unsigned int i = 0; i < type_mapping.size(); i++)
            strncpy(&types[max_len * i], type_mapping[i].c_str(), max_len);
        int retval = gsd_write_chunk(&m_handle,
                                     chunk.c_str(),
                                     GSD_TYPE_UINT8,
                                     type_mapping.size(),
                                     max_len,
                                     0,
                                     (void*)&types[0]);
        GSDUtils::checkError(retval, m_fname);
        }
    }

/*! \param timestep

    Write the data chunks configuration/step, configuration/box, and particles/N. If this is frame
   0, also write configuration/dimensions.

    N is not strictly necessary for constant N data, but is always written in case the user fails to
   select dynamic attributes with a variable N file.
*/
void GSDDumpWriter::writeFrameHeader(uint64_t timestep)
    {
    int retval;
    m_exec_conf->msg->notice(10) << "GSD: writing configuration/step" << endl;
    uint64_t step = timestep;
    retval
        = gsd_write_chunk(&m_handle, "configuration/step", GSD_TYPE_UINT64, 1, 1, 0, (void*)&step);
    GSDUtils::checkError(retval, m_fname);

    if (gsd_get_nframes(&m_handle) == 0)
        {
        m_exec_conf->msg->notice(10) << "GSD: writing configuration/dimensions" << endl;
        uint8_t dimensions = (uint8_t)m_sysdef->getNDimensions();
        retval = gsd_write_chunk(&m_handle,
                                 "configuration/dimensions",
                                 GSD_TYPE_UINT8,
                                 1,
                                 1,
                                 0,
                                 (void*)&dimensions);
        GSDUtils::checkError(retval, m_fname);
        }

    m_exec_conf->msg->notice(10) << "GSD: writing configuration/box" << endl;
    BoxDim box = m_pdata->getGlobalBox();
    float box_a[6];
    box_a[0] = (float)box.getL().x;
    box_a[1] = (float)box.getL().y;
    box_a[2] = (float)box.getL().z;
    box_a[3] = (float)box.getTiltFactorXY();
    box_a[4] = (float)box.getTiltFactorXZ();
    box_a[5] = (float)box.getTiltFactorYZ();
    retval = gsd_write_chunk(&m_handle, "configuration/box", GSD_TYPE_FLOAT, 6, 1, 0, (void*)box_a);
    GSDUtils::checkError(retval, m_fname);

    m_exec_conf->msg->notice(10) << "GSD: writing particles/N" << endl;
    uint32_t N = m_group->getNumMembersGlobal();
    retval = gsd_write_chunk(&m_handle, "particles/N", GSD_TYPE_UINT32, 1, 1, 0, (void*)&N);
    GSDUtils::checkError(retval, m_fname);
    }

/*! \param snapshot particle data snapshot to write out to the file

    Writes the data chunks types, typeid, mass, charge, diameter, body, moment_inertia in
   particles/.
*/
void GSDDumpWriter::writeAttributes(const SnapshotParticleData<float>& snapshot,
                                    const std::map<unsigned int, unsigned int>& map)
    {
    uint32_t N = m_group->getNumMembersGlobal();
    int retval;
    uint64_t nframes = gsd_get_nframes(&m_handle);

    writeTypeMapping("particles/types", snapshot.type_mapping);

        {
        std::vector<uint32_t> type(N);
        type.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.type[it->second] != 0)
                all_default = false;

            type[group_idx] = uint32_t(snapshot.type[it->second]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/typeid"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/typeid" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/typeid",
                                     GSD_TYPE_UINT32,
                                     N,
                                     1,
                                     0,
                                     (void*)&type[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/typeid"] = true;
            }
        }

        {
        std::vector<float> data(N);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.mass[it->second] != float(1.0))
                all_default = false;

            data[group_idx] = float(snapshot.mass[it->second]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/mass"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/mass" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/mass",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     1,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/mass"] = true;
            }

        all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.charge[it->second] != float(0.0))
                all_default = false;
            data[group_idx] = float(snapshot.charge[it->second]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/charge"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/charge" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/charge",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     1,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/charge"] = true;
            }

        all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.diameter[it->second] != float(1.0))
                all_default = false;

            data[group_idx] = float(snapshot.diameter[it->second]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/diameter"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/diameter" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/diameter",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     1,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/diameter"] = true;
            }
        }

        {
        std::vector<int32_t> body(N);
        body.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.body[it->second] != NO_BODY)
                all_default = false;

            body[group_idx] = int32_t(snapshot.body[it->second]);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/body"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/body" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/body",
                                     GSD_TYPE_INT32,
                                     N,
                                     1,
                                     0,
                                     (void*)&body[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/body"] = true;
            }
        }

        {
        std::vector<float> data(uint64_t(N) * 3);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.inertia[it->second].x != float(0.0)
                || snapshot.inertia[it->second].y != float(0.0)
                || snapshot.inertia[it->second].z != float(0.0))
                {
                all_default = false;
                }

            data[group_idx * 3 + 0] = float(snapshot.inertia[it->second].x);
            data[group_idx * 3 + 1] = float(snapshot.inertia[it->second].y);
            data[group_idx * 3 + 2] = float(snapshot.inertia[it->second].z);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/moment_inertia"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/moment_inertia" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/moment_inertia",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     3,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/moment_inertia"] = true;
            }
        }
    }

/*! \param snapshot particle data snapshot to write out to the file

    Writes the data chunks position and orientation in particles/.
*/
void GSDDumpWriter::writeProperties(const SnapshotParticleData<float>& snapshot,
                                    const std::map<unsigned int, unsigned int>& map)
    {
    uint32_t N = m_group->getNumMembersGlobal();
    int retval;
    uint64_t nframes = gsd_get_nframes(&m_handle);

        {
        std::vector<float> data(uint64_t(N) * 3);
        data.reserve(1); //! make sure we allocate

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            data[group_idx * 3 + 0] = float(snapshot.pos[it->second].x);
            data[group_idx * 3 + 1] = float(snapshot.pos[it->second].y);
            data[group_idx * 3 + 2] = float(snapshot.pos[it->second].z);
            }

        m_exec_conf->msg->notice(10) << "GSD: writing particles/position" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/position",
                                 GSD_TYPE_FLOAT,
                                 N,
                                 3,
                                 0,
                                 (void*)&data[0]);
        GSDUtils::checkError(retval, m_fname);
        }

        {
        std::vector<float> data(uint64_t(N) * 4);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.orientation[it->second].s != float(1.0)
                || snapshot.orientation[it->second].v.x != float(0.0)
                || snapshot.orientation[it->second].v.y != float(0.0)
                || snapshot.orientation[it->second].v.z != float(0.0))
                {
                all_default = false;
                }

            data[group_idx * 4 + 0] = float(snapshot.orientation[it->second].s);
            data[group_idx * 4 + 1] = float(snapshot.orientation[it->second].v.x);
            data[group_idx * 4 + 2] = float(snapshot.orientation[it->second].v.y);
            data[group_idx * 4 + 3] = float(snapshot.orientation[it->second].v.z);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/orientation"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/orientation" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/orientation",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     4,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/orientation"] = true;
            }
        }
    }

/*! \param snapshot particle data snapshot to write out to the file

    Writes the data chunks velocity, angmom, and image in particles/.
*/
void GSDDumpWriter::writeMomenta(const SnapshotParticleData<float>& snapshot,
                                 const std::map<unsigned int, unsigned int>& map)
    {
    uint32_t N = m_group->getNumMembersGlobal();
    int retval;
    uint64_t nframes = gsd_get_nframes(&m_handle);

        {
        std::vector<float> data(uint64_t(N) * 3);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.vel[it->second].x != float(0.0) || snapshot.vel[it->second].y != float(0.0)
                || snapshot.vel[it->second].z != float(0.0))
                {
                all_default = false;
                }

            data[group_idx * 3 + 0] = float(snapshot.vel[it->second].x);
            data[group_idx * 3 + 1] = float(snapshot.vel[it->second].y);
            data[group_idx * 3 + 2] = float(snapshot.vel[it->second].z);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/velocity"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/velocity" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/velocity",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     3,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/velocity"] = true;
            }
        }

        {
        std::vector<float> data(uint64_t(N) * 4);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.angmom[it->second].s != float(0.0)
                || snapshot.angmom[it->second].v.x != float(0.0)
                || snapshot.angmom[it->second].v.y != float(0.0)
                || snapshot.angmom[it->second].v.z != float(0.0))
                {
                all_default = false;
                }

            data[group_idx * 4 + 0] = float(snapshot.angmom[it->second].s);
            data[group_idx * 4 + 1] = float(snapshot.angmom[it->second].v.x);
            data[group_idx * 4 + 2] = float(snapshot.angmom[it->second].v.y);
            data[group_idx * 4 + 3] = float(snapshot.angmom[it->second].v.z);
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/angmom"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/angmom" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/angmom",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     4,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/angmom"] = true;
            }
        }

        {
        std::vector<int32_t> data(uint64_t(N) * 3);
        data.reserve(1); //! make sure we allocate
        bool all_default = true;

        for (unsigned int group_idx = 0; group_idx < N; group_idx++)
            {
            unsigned int t = m_group->getMemberTag(group_idx);

            // look up tag in snapshot
            auto it = map.find(t);
            assert(it != map.end());

            if (snapshot.image[it->second].x != 0 || snapshot.image[it->second].y != 0
                || snapshot.image[it->second].z != 0)
                {
                all_default = false;
                }

            data[group_idx * 3 + 0] = snapshot.image[it->second].x;
            data[group_idx * 3 + 1] = snapshot.image[it->second].y;
            data[group_idx * 3 + 2] = snapshot.image[it->second].z;
            }

        if (!all_default || (nframes > 0 && m_nondefault["particles/image"]))
            {
            m_exec_conf->msg->notice(10) << "GSD: writing particles/image" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/image",
                                     GSD_TYPE_INT32,
                                     N,
                                     3,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            if (nframes == 0)
                m_nondefault["particles/image"] = true;
            }
        }
    }

/*! \param bond Bond data snapshot
    \param angle Angle data snapshot
    \param dihedral Dihedral data snapshot
    \param improper Improper data snapshot
    \param constraint Constraint data snapshot
    \param pair Special pair data snapshot

    Write out all the snapshot data to the GSD file
*/
void GSDDumpWriter::writeTopology(BondData::Snapshot& bond,
                                  AngleData::Snapshot& angle,
                                  DihedralData::Snapshot& dihedral,
                                  ImproperData::Snapshot& improper,
                                  ConstraintData::Snapshot& constraint,
                                  PairData::Snapshot& pair)
    {
    if (bond.size > 0)
        {
        m_exec_conf->msg->notice(10) << "GSD: writing bonds/N" << endl;
        uint32_t N = bond.size;
        int retval = gsd_write_chunk(&m_handle, "bonds/N", GSD_TYPE_UINT32, 1, 1, 0, (void*)&N);
        GSDUtils::checkError(retval, m_fname);

        writeTypeMapping("bonds/types", bond.type_mapping);

        m_exec_conf->msg->notice(10) << "GSD: writing bonds/typeid" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "bonds/typeid",
                                 GSD_TYPE_UINT32,
                                 N,
                                 1,
                                 0,
                                 (void*)&bond.type_id[0]);
        GSDUtils::checkError(retval, m_fname);

        m_exec_conf->msg->notice(10) << "GSD: writing bonds/group" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "bonds/group",
                                 GSD_TYPE_UINT32,
                                 N,
                                 2,
                                 0,
                                 (void*)&bond.groups[0]);
        GSDUtils::checkError(retval, m_fname);
        }
    if (angle.size > 0)
        {
        m_exec_conf->msg->notice(10) << "GSD: writing angles/N" << endl;
        uint32_t N = angle.size;
        int retval = gsd_write_chunk(&m_handle, "angles/N", GSD_TYPE_UINT32, 1, 1, 0, (void*)&N);
        GSDUtils::checkError(retval, m_fname);

        writeTypeMapping("angles/types", angle.type_mapping);

        m_exec_conf->msg->notice(10) << "GSD: writing angles/typeid" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "angles/typeid",
                                 GSD_TYPE_UINT32,
                                 N,
                                 1,
                                 0,
                                 (void*)&angle.type_id[0]);
        GSDUtils::checkError(retval, m_fname);

        m_exec_conf->msg->notice(10) << "GSD: writing angles/group" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "angles/group",
                                 GSD_TYPE_UINT32,
                                 N,
                                 3,
                                 0,
                                 (void*)&angle.groups[0]);
        GSDUtils::checkError(retval, m_fname);
        }
    if (dihedral.size > 0)
        {
        m_exec_conf->msg->notice(10) << "GSD: writing dihedrals/N" << endl;
        uint32_t N = dihedral.size;
        int retval = gsd_write_chunk(&m_handle, "dihedrals/N", GSD_TYPE_UINT32, 1, 1, 0, (void*)&N);
        GSDUtils::checkError(retval, m_fname);

        writeTypeMapping("dihedrals/types", dihedral.type_mapping);

        m_exec_conf->msg->notice(10) << "GSD: writing dihedrals/typeid" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "dihedrals/typeid",
                                 GSD_TYPE_UINT32,
                                 N,
                                 1,
                                 0,
                                 (void*)&dihedral.type_id[0]);
        GSDUtils::checkError(retval, m_fname);

        m_exec_conf->msg->notice(10) << "GSD: writing dihedrals/group" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "dihedrals/group",
                                 GSD_TYPE_UINT32,
                                 N,
                                 4,
                                 0,
                                 (void*)&dihedral.groups[0]);
        GSDUtils::checkError(retval, m_fname);
        }
    if (improper.size > 0)
        {
        m_exec_conf->msg->notice(10) << "GSD: writing impropers/N" << endl;
        uint32_t N = improper.size;
        int retval = gsd_write_chunk(&m_handle, "impropers/N", GSD_TYPE_UINT32, 1, 1, 0, (void*)&N);
        GSDUtils::checkError(retval, m_fname);

        writeTypeMapping("impropers/types", improper.type_mapping);

        m_exec_conf->msg->notice(10) << "GSD: writing impropers/typeid" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "impropers/typeid",
                                 GSD_TYPE_UINT32,
                                 N,
                                 1,
                                 0,
                                 (void*)&improper.type_id[0]);
        GSDUtils::checkError(retval, m_fname);

        m_exec_conf->msg->notice(10) << "GSD: writing impropers/group" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "impropers/group",
                                 GSD_TYPE_UINT32,
                                 N,
                                 4,
                                 0,
                                 (void*)&improper.groups[0]);
        GSDUtils::checkError(retval, m_fname);
        }

    if (constraint.size > 0)
        {
        m_exec_conf->msg->notice(10) << "GSD: writing constraints/N" << endl;
        uint32_t N = constraint.size;
        int retval
            = gsd_write_chunk(&m_handle, "constraints/N", GSD_TYPE_UINT32, 1, 1, 0, (void*)&N);
        GSDUtils::checkError(retval, m_fname);

        m_exec_conf->msg->notice(10) << "GSD: writing constraints/value" << endl;
            {
            std::vector<float> data(N);
            data.reserve(1); //! make sure we allocate
            for (unsigned int i = 0; i < N; i++)
                data[i] = float(constraint.val[i]);

            retval = gsd_write_chunk(&m_handle,
                                     "constraints/value",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     1,
                                     0,
                                     (void*)&data[0]);
            GSDUtils::checkError(retval, m_fname);
            }

        m_exec_conf->msg->notice(10) << "GSD: writing constraints/group" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "constraints/group",
                                 GSD_TYPE_UINT32,
                                 N,
                                 2,
                                 0,
                                 (void*)&constraint.groups[0]);
        GSDUtils::checkError(retval, m_fname);
        }

    if (pair.size > 0)
        {
        m_exec_conf->msg->notice(10) << "GSD: writing pairs/N" << endl;
        uint32_t N = pair.size;
        int retval = gsd_write_chunk(&m_handle, "pairs/N", GSD_TYPE_UINT32, 1, 1, 0, (void*)&N);
        GSDUtils::checkError(retval, m_fname);

        writeTypeMapping("pairs/types", pair.type_mapping);

        m_exec_conf->msg->notice(10) << "GSD: writing pairs/typeid" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "pairs/typeid",
                                 GSD_TYPE_UINT32,
                                 N,
                                 1,
                                 0,
                                 (void*)&pair.type_id[0]);
        GSDUtils::checkError(retval, m_fname);

        m_exec_conf->msg->notice(10) << "GSD: writing pairs/group" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "pairs/group",
                                 GSD_TYPE_UINT32,
                                 N,
                                 2,
                                 0,
                                 (void*)&pair.groups[0]);
        GSDUtils::checkError(retval, m_fname);
        }
    }

void GSDDumpWriter::writeLogQuantities(pybind11::dict dict)
    {
    bool root = true;
#ifdef ENABLE_MPI
    root = m_exec_conf->isRoot();
#endif

    // only evaluate the numpy array on the root rank
    if (root)
        {
        for (auto key_iter = dict.begin(); key_iter != dict.end(); ++key_iter)
            {
            std::string name = pybind11::cast<std::string>(key_iter->first);
            m_exec_conf->msg->notice(10) << "GSD: writing " << name << endl;

            pybind11::array arr
                = pybind11::array::ensure(key_iter->second, pybind11::array::c_style);
            gsd_type type = GSD_TYPE_UINT8;
            auto dtype = arr.dtype();
            if (dtype.kind() == 'u' && dtype.itemsize() == 1)
                {
                type = GSD_TYPE_UINT8;
                }
            else if (dtype.kind() == 'u' && dtype.itemsize() == 2)
                {
                type = GSD_TYPE_UINT16;
                }
            else if (dtype.kind() == 'u' && dtype.itemsize() == 4)
                {
                type = GSD_TYPE_UINT32;
                }
            else if (dtype.kind() == 'u' && dtype.itemsize() == 8)
                {
                type = GSD_TYPE_UINT64;
                }
            else if (dtype.kind() == 'i' && dtype.itemsize() == 1)
                {
                type = GSD_TYPE_INT8;
                }
            else if (dtype.kind() == 'i' && dtype.itemsize() == 2)
                {
                type = GSD_TYPE_INT16;
                }
            else if (dtype.kind() == 'i' && dtype.itemsize() == 4)
                {
                type = GSD_TYPE_INT32;
                }
            else if (dtype.kind() == 'i' && dtype.itemsize() == 8)
                {
                type = GSD_TYPE_INT64;
                }
            else if (dtype.kind() == 'f' && dtype.itemsize() == 4)
                {
                type = GSD_TYPE_FLOAT;
                }
            else if (dtype.kind() == 'f' && dtype.itemsize() == 8)
                {
                type = GSD_TYPE_DOUBLE;
                }
            else if (dtype.kind() == 'b' && dtype.itemsize() == 1)
                {
                type = GSD_TYPE_UINT8;
                }
            else
                {
                throw range_error("Invalid numpy array format in gsd log data [" + name
                                  + "]: " + string(pybind11::str(arr.dtype())));
                }

            size_t M = 1;
            size_t N = 1;
            auto ndim = arr.ndim();
            if (ndim == 0)
                {
                // numpy converts scalars to arrays with zero dimensions
                // gsd treats them as 1x1 arrays.
                M = 1;
                N = 1;
                }
            if (ndim == 1)
                {
                N = arr.shape(0);
                M = 1;
                }
            if (ndim == 2)
                {
                N = arr.shape(0);
                M = arr.shape(1);
                if (M > std::numeric_limits<uint32_t>::max())
                    throw runtime_error("Array dimension too large in gsd log data [" + name + "]");
                }
            if (ndim > 2)
                {
                throw invalid_argument("Invalid numpy dimension in gsd log data [" + name + "]");
                }

            int retval = gsd_write_chunk(&m_handle,
                                         name.c_str(),
                                         type,
                                         N,
                                         (uint32_t)M,
                                         0,
                                         (void*)arr.data());
            GSDUtils::checkError(retval, m_fname);
            }
        }
    }

/*! Populate the m_nondefault map.
    Set entries to true when they exist in frame 0 of the file, otherwise, set them to false.
*/
void GSDDumpWriter::populateNonDefault()
    {
    int retval;

    // open the file in read only mode
    m_exec_conf->msg->notice(3) << "GSD: check frame 0 in gsd file " << m_fname << endl;
    retval = gsd_open(&m_handle, m_fname.c_str(), GSD_OPEN_READONLY);
    GSDUtils::checkError(retval, m_fname);

    // validate schema
    if (string(m_handle.header.schema) != string("hoomd"))
        {
        std::ostringstream s;
        s << "GSD: "
          << "Invalid schema in " << m_fname;
        throw runtime_error("Error opening GSD file");
        }
    if (m_handle.header.schema_version >= gsd_make_version(2, 0))
        {
        std::ostringstream s;
        s << "GSD: "
          << "Invalid schema version in " << m_fname;
        throw runtime_error("Error opening GSD file");
        }

    for (auto const& chunk : particle_chunks)
        {
        const gsd_index_entry* entry = gsd_find_chunk(&m_handle, 0, chunk.c_str());
        m_nondefault[chunk] = (entry != nullptr);
        }

    // close the file
    gsd_close(&m_handle);
    }

namespace detail
    {
void export_GSDDumpWriter(pybind11::module& m)
    {
    pybind11::bind_map<std::map<std::string, pybind11::function>>(m, "MapStringFunction");

    pybind11::class_<GSDDumpWriter, Analyzer, std::shared_ptr<GSDDumpWriter>>(m, "GSDDumpWriter")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::string,
                            std::shared_ptr<ParticleGroup>,
                            std::string,
                            bool>())
        .def("setWriteAttribute", &GSDDumpWriter::setWriteAttribute)
        .def("setWriteProperty", &GSDDumpWriter::setWriteProperty)
        .def("setWriteMomentum", &GSDDumpWriter::setWriteMomentum)
        .def("setWriteTopology", &GSDDumpWriter::setWriteTopology)
        .def("writeLogQuantities", &GSDDumpWriter::writeLogQuantities)
        .def_property("log_writer", &GSDDumpWriter::getLogWriter, &GSDDumpWriter::setLogWriter)
        .def_property_readonly("filename", &GSDDumpWriter::getFilename)
        .def_property_readonly("mode", &GSDDumpWriter::getMode)
        .def_property_readonly("dynamic", &GSDDumpWriter::getDynamic)
        .def_property_readonly("truncate", &GSDDumpWriter::getTruncate)
        .def_property_readonly("filter",
                               [](const std::shared_ptr<GSDDumpWriter> gsd)
                               { return gsd->getGroup()->getFilter(); });
    }

    } // end namespace detail

    } // end namespace hoomd
