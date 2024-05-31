// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
std::list<std::string> GSDDumpWriter::particle_chunks {"particles/position",
                                                       "particles/typeid",
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
                             std::shared_ptr<Trigger> trigger,
                             const std::string& fname,
                             std::shared_ptr<ParticleGroup> group,
                             std::string mode,
                             bool truncate)
    : Analyzer(sysdef, trigger), m_fname(fname), m_mode(mode), m_truncate(truncate), m_group(group)
    {
    m_exec_conf->msg->notice(5) << "Constructing GSDDumpWriter: " << m_fname << " " << mode << " "
                                << truncate << endl;
    if (mode != "wb" && mode != "xb" && mode != "ab")
        {
        throw std::invalid_argument("Invalid GSD file mode: " + mode);
        }
    m_log_writer = pybind11::none();

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        m_gather_tag_order = GatherTagOrder(m_exec_conf->getMPICommunicator());
        }
#endif

    m_dynamic.reset();
    m_dynamic[gsd_flag::particles_position] = true;
    m_dynamic[gsd_flag::particles_orientation] = true;

    initFileIO();
    }

pybind11::tuple GSDDumpWriter::getDynamic()
    {
    pybind11::list result;

    if (m_dynamic[gsd_flag::configuration_box])
        {
        result.append("configuration/box");
        }
    if (m_dynamic[gsd_flag::particles_N])
        {
        result.append("particles/N");
        }
    if (m_dynamic[gsd_flag::particles_position])
        {
        result.append("particles/position");
        }
    if (m_dynamic[gsd_flag::particles_orientation])
        {
        result.append("particles/orientation");
        }
    if (m_dynamic[gsd_flag::particles_velocity])
        {
        result.append("particles/velocity");
        }
    if (m_dynamic[gsd_flag::particles_angmom])
        {
        result.append("particles/angmom");
        }
    if (m_dynamic[gsd_flag::particles_image])
        {
        result.append("particles/image");
        }
    if (m_dynamic[gsd_flag::particles_types])
        {
        result.append("particles/types");
        }
    if (m_dynamic[gsd_flag::particles_type])
        {
        result.append("particles/typeid");
        }
    if (m_dynamic[gsd_flag::particles_mass])
        {
        result.append("particles/mass");
        }
    if (m_dynamic[gsd_flag::particles_charge])
        {
        result.append("particles/charge");
        }
    if (m_dynamic[gsd_flag::particles_diameter])
        {
        result.append("particles/diameter");
        }
    if (m_dynamic[gsd_flag::particles_body])
        {
        result.append("particles/body");
        }
    if (m_dynamic[gsd_flag::particles_inertia])
        {
        result.append("particles/moment_inertia");
        }
    if (m_write_topology)
        {
        result.append("topology");
        }

    return pybind11::tuple(result);
    }

void GSDDumpWriter::setDynamic(pybind11::object dynamic)
    {
    pybind11::list dynamic_list = dynamic;
    m_dynamic.reset();
    m_write_topology = false;

    for (const auto& s_py : dynamic_list)
        {
        std::string s = s_py.cast<std::string>();
        if (s == "configuration/box" || s == "property")
            {
            m_dynamic[gsd_flag::configuration_box] = true;
            }
        if (s == "particles/N" || s == "property")
            {
            m_dynamic[gsd_flag::particles_N] = true;
            }
        if (s == "particles/position" || s == "property")
            {
            m_dynamic[gsd_flag::particles_position] = true;
            }
        if (s == "particles/orientation" || s == "property")
            {
            m_dynamic[gsd_flag::particles_orientation] = true;
            }
        if (s == "particles/velocity" || s == "momentum")
            {
            m_dynamic[gsd_flag::particles_velocity] = true;
            }
        if (s == "particles/angmom" || s == "momentum")
            {
            m_dynamic[gsd_flag::particles_angmom] = true;
            }
        if (s == "particles/image" || s == "momentum")
            {
            m_dynamic[gsd_flag::particles_image] = true;
            }
        if (s == "particles/types" || s == "attribute")
            {
            m_dynamic[gsd_flag::particles_types] = true;
            }
        if (s == "particles/typeid" || s == "attribute")
            {
            m_dynamic[gsd_flag::particles_type] = true;
            }
        if (s == "particles/mass" || s == "attribute")
            {
            m_dynamic[gsd_flag::particles_mass] = true;
            }
        if (s == "particles/charge" || s == "attribute")
            {
            m_dynamic[gsd_flag::particles_charge] = true;
            }
        if (s == "particles/diameter" || s == "attribute")
            {
            m_dynamic[gsd_flag::particles_diameter] = true;
            }
        if (s == "particles/body" || s == "attribute")
            {
            m_dynamic[gsd_flag::particles_body] = true;
            }
        if (s == "particles/moment_inertia" || s == "attribute")
            {
            m_dynamic[gsd_flag::particles_inertia] = true;
            }
        if (s == "topology")
            {
            m_write_topology = true;
            }
        }
    }

void GSDDumpWriter::flush()
    {
    if (m_exec_conf->isRoot())
        {
        m_exec_conf->msg->notice(5) << "GSD: flush gsd file " << m_fname << endl;
        int retval = gsd_flush(&m_handle);
        GSDUtils::checkError(retval, m_fname);
        }
    }

void GSDDumpWriter::setMaximumWriteBufferSize(uint64_t size)
    {
    if (m_exec_conf->isRoot())
        {
        int retval = gsd_set_maximum_write_buffer_size(&m_handle, size);
        GSDUtils::checkError(retval, m_fname);

        // Scale the index buffer entires to write with the write buffer.
        retval = gsd_set_index_entries_to_buffer(&m_handle, size / 256);
        GSDUtils::checkError(retval, m_fname);
        }
    }

uint64_t GSDDumpWriter::getMaximumWriteBufferSize()
    {
    if (m_exec_conf->isRoot())
        {
        return gsd_get_maximum_write_buffer_size(&m_handle);
        }
    else
        {
        return 0;
        }
    }

//! Initializes the output file for writing
void GSDDumpWriter::initFileIO()
    {
    if (m_exec_conf->isRoot())
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
                s << "GSD: " << "Invalid schema in " << m_fname;
                throw runtime_error("Error opening GSD file");
                }
            if (m_handle.header.schema_version >= gsd_make_version(2, 0))
                {
                std::ostringstream s;
                s << "GSD: " << "Invalid schema version in " << m_fname;
                throw runtime_error("Error opening GSD file");
                }
            }
        else
            {
            throw std::invalid_argument("Invalid GSD file mode: " + m_mode);
            }

        m_nframes = gsd_get_nframes(&m_handle);
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        bcast(m_nframes, 0, m_exec_conf->getMPICommunicator());
        bcast(m_nondefault, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    }

GSDDumpWriter::~GSDDumpWriter()
    {
    m_exec_conf->msg->notice(5) << "Destroying GSDDumpWriter" << endl;

    if (m_exec_conf->isRoot())
        {
        m_exec_conf->msg->notice(5) << "GSD: close gsd file " << m_fname << endl;
        gsd_close(&m_handle);
        }
    }

//! Get the logged data for the current frame if any.
pybind11::dict GSDDumpWriter::getLogData() const
    {
    if (!m_log_writer.is_none())
        {
        return m_log_writer.attr("log")().cast<pybind11::dict>();
        }
    return pybind11::dict();
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

    // truncate the file if requested
    if (m_truncate)
        {
        if (m_exec_conf->isRoot())
            {
            m_exec_conf->msg->notice(10) << "GSD: truncating file" << endl;
            retval = gsd_truncate(&m_handle);
            GSDUtils::checkError(retval, m_fname);
            }

        m_nframes = 0;
        }

    populateLocalFrame(m_local_frame, timestep);
    auto log_data = getLogData();
    write(m_local_frame, log_data);
    }

void GSDDumpWriter::write(GSDDumpWriter::GSDFrame& frame, pybind11::dict log_data)
    {
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        gatherGlobalFrame(frame);

        if (m_exec_conf->isRoot())
            {
            writeFrameHeader(m_global_frame);
            writeAttributes(m_global_frame);
            writeProperties(m_global_frame);
            writeMomenta(m_global_frame);
            writeLogQuantities(log_data);
            }
        }
    else
#endif
        {
        writeFrameHeader(frame);
        writeAttributes(frame);
        writeProperties(frame);
        writeMomenta(frame);
        writeLogQuantities(log_data);
        }
    // topology is only meaningful if this is the all group
    if (m_group->getNumMembersGlobal() == m_pdata->getNGlobal()
        && (m_write_topology || m_nframes == 0))
        {
        if (m_exec_conf->isRoot())
            {
            writeTopology(frame.bond_data,
                          frame.angle_data,
                          frame.dihedral_data,
                          frame.improper_data,
                          frame.constraint_data,
                          frame.pair_data);
            }
        }

    if (m_exec_conf->isRoot())
        {
        m_exec_conf->msg->notice(10) << "GSD: ending frame" << endl;
        int retval = gsd_end_frame(&m_handle);
        GSDUtils::checkError(retval, m_fname);
        }

    m_nframes++;
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

/*! Write the data chunks configuration/step, configuration/box, and particles/N. If this is frame
   0, also write configuration/dimensions.
*/
void GSDDumpWriter::writeFrameHeader(const GSDDumpWriter::GSDFrame& frame)
    {
    int retval;
    m_exec_conf->msg->notice(10) << "GSD: writing configuration/step" << endl;
    retval = gsd_write_chunk(&m_handle,
                             "configuration/step",
                             GSD_TYPE_UINT64,
                             1,
                             1,
                             0,
                             (void*)&frame.timestep);
    GSDUtils::checkError(retval, m_fname);

    if (m_nframes == 0)
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

    if (m_nframes == 0 || m_dynamic[gsd_flag::configuration_box])
        {
        m_exec_conf->msg->notice(10) << "GSD: writing configuration/box" << endl;
        float box_a[6];
        box_a[0] = (float)frame.global_box.getL().x;
        box_a[1] = (float)frame.global_box.getL().y;
        box_a[2] = (float)frame.global_box.getL().z;
        box_a[3] = (float)frame.global_box.getTiltFactorXY();
        box_a[4] = (float)frame.global_box.getTiltFactorXZ();
        box_a[5] = (float)frame.global_box.getTiltFactorYZ();
        retval = gsd_write_chunk(&m_handle,
                                 "configuration/box",
                                 GSD_TYPE_FLOAT,
                                 6,
                                 1,
                                 0,
                                 (void*)box_a);
        GSDUtils::checkError(retval, m_fname);
        }

    if (m_nframes == 0 || m_dynamic[gsd_flag::particles_N])
        {
        m_exec_conf->msg->notice(10) << "GSD: writing particles/N" << endl;
        uint32_t N = m_group->getNumMembersGlobal();
        retval = gsd_write_chunk(&m_handle, "particles/N", GSD_TYPE_UINT32, 1, 1, 0, (void*)&N);
        GSDUtils::checkError(retval, m_fname);
        }
    }

/*! Writes the data chunks types, typeid, mass, charge, diameter, body, moment_inertia in
   particles/.
*/
void GSDDumpWriter::writeAttributes(const GSDDumpWriter::GSDFrame& frame)
    {
    uint32_t N = m_group->getNumMembersGlobal();
    int retval;

    if (m_dynamic[gsd_flag::particles_types] || m_nframes == 0)
        {
        writeTypeMapping("particles/types", frame.particle_data.type_mapping);
        }

    if (frame.particle_data.type.size() != 0)
        {
        assert(frame.particle_data.type.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/typeid" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/typeid",
                                 GSD_TYPE_UINT32,
                                 N,
                                 1,
                                 0,
                                 (void*)frame.particle_data.type.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/typeid"] = true;
        }

    if (frame.particle_data.mass.size() != 0)
        {
        assert(frame.particle_data.mass.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/mass" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/mass",
                                 GSD_TYPE_FLOAT,
                                 N,
                                 1,
                                 0,
                                 (void*)frame.particle_data.mass.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/mass"] = true;
        }

    if (frame.particle_data.charge.size() != 0)
        {
        assert(frame.particle_data.charge.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/charge" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/charge",
                                 GSD_TYPE_FLOAT,
                                 N,
                                 1,
                                 0,
                                 (void*)frame.particle_data.charge.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/charge"] = true;
        }

    if (m_write_diameter)
        {
        if (frame.particle_data.diameter.size() != 0)
            {
            assert(frame.particle_data.diameter.size() == N);

            m_exec_conf->msg->notice(10) << "GSD: writing particles/diameter" << endl;
            retval = gsd_write_chunk(&m_handle,
                                     "particles/diameter",
                                     GSD_TYPE_FLOAT,
                                     N,
                                     1,
                                     0,
                                     (void*)frame.particle_data.diameter.data());
            GSDUtils::checkError(retval, m_fname);
            if (m_nframes == 0)
                m_nondefault["particles/diameter"] = true;
            }
        }

    if (frame.particle_data.body.size() != 0)
        {
        assert(frame.particle_data.body.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/body" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/body",
                                 GSD_TYPE_INT32,
                                 N,
                                 1,
                                 0,
                                 (void*)frame.particle_data.body.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/body"] = true;
        }

    if (frame.particle_data.inertia.size() != 0)
        {
        assert(frame.particle_data.inertia.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/moment_inertia" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/moment_inertia",
                                 GSD_TYPE_FLOAT,
                                 N,
                                 3,
                                 0,
                                 (void*)frame.particle_data.inertia.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/moment_inertia"] = true;
        }
    }

/*! Writes the data chunks position and orientation in particles/.
 */
void GSDDumpWriter::writeProperties(const GSDDumpWriter::GSDFrame& frame)
    {
    uint32_t N = m_group->getNumMembersGlobal();
    int retval;

    if (frame.particle_data.pos.size() != 0)
        {
        assert(frame.particle_data.pos.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/position" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/position",
                                 GSD_TYPE_FLOAT,
                                 N,
                                 3,
                                 0,
                                 (void*)frame.particle_data.pos.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/position"] = true;
        }

    if (frame.particle_data.orientation.size() != 0)
        {
        assert(frame.particle_data.orientation.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/orientation" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/orientation",
                                 GSD_TYPE_FLOAT,
                                 N,
                                 4,
                                 0,
                                 (void*)frame.particle_data.orientation.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/orientation"] = true;
        }
    }

/*! Writes the data chunks velocity, angmom, and image in particles/.
 */
void GSDDumpWriter::writeMomenta(const GSDDumpWriter::GSDFrame& frame)
    {
    uint32_t N = m_group->getNumMembersGlobal();
    int retval;

    if (frame.particle_data.vel.size() != 0)
        {
        assert(frame.particle_data.vel.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/velocity" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/velocity",
                                 GSD_TYPE_FLOAT,
                                 N,
                                 3,
                                 0,
                                 (void*)frame.particle_data.vel.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/velocity"] = true;
        }

    if (frame.particle_data.angmom.size() != 0)
        {
        assert(frame.particle_data.angmom.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/angmom" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/angmom",
                                 GSD_TYPE_FLOAT,
                                 N,
                                 4,
                                 0,
                                 (void*)frame.particle_data.angmom.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/angmom"] = true;
        }

    if (frame.particle_data.image.size() != 0)
        {
        assert(frame.particle_data.image.size() == N);

        m_exec_conf->msg->notice(10) << "GSD: writing particles/image" << endl;
        retval = gsd_write_chunk(&m_handle,
                                 "particles/image",
                                 GSD_TYPE_INT32,
                                 N,
                                 3,
                                 0,
                                 (void*)frame.particle_data.image.data());
        GSDUtils::checkError(retval, m_fname);
        if (m_nframes == 0)
            m_nondefault["particles/image"] = true;
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
    for (auto key_iter = dict.begin(); key_iter != dict.end(); ++key_iter)
        {
        std::string name = pybind11::cast<std::string>(key_iter->first);
        m_exec_conf->msg->notice(10) << "GSD: writing " << name << endl;

        pybind11::array arr = pybind11::array::ensure(key_iter->second, pybind11::array::c_style);
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

        int retval
            = gsd_write_chunk(&m_handle, name.c_str(), type, N, (uint32_t)M, 0, (void*)arr.data());
        GSDUtils::checkError(retval, m_fname);
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
        s << "GSD: " << "Invalid schema in " << m_fname;
        throw runtime_error("Error opening GSD file");
        }
    if (m_handle.header.schema_version >= gsd_make_version(2, 0))
        {
        std::ostringstream s;
        s << "GSD: " << "Invalid schema version in " << m_fname;
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

void GSDDumpWriter::populateLocalFrame(GSDDumpWriter::GSDFrame& frame, uint64_t timestep)
    {
    frame.timestep = timestep;
    frame.global_box = m_pdata->getGlobalBox();

    frame.particle_data.type_mapping = m_pdata->getTypeMapping();

    uint32_t N = m_group->getNumMembersGlobal();

    // Assume values are all default to start, set flags to false when we find a non-default.
    std::bitset<n_gsd_flags> all_default;
    all_default.set();
    frame.clear();

    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    if (N > 0)
        {
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);

        m_index.resize(0);

        for (unsigned int group_tag_index = 0; group_tag_index < N; group_tag_index++)
            {
            unsigned int tag = m_group->getMemberTag(group_tag_index);
            unsigned int index = h_rtag.data[tag];
            if (index >= m_pdata->getN())
                {
                continue;
                }

            frame.particle_tags.push_back(h_tag.data[index]);
            m_index.push_back(index);
            }
        }

    if (N > 0
        && (m_dynamic[gsd_flag::particles_position] || m_dynamic[gsd_flag::particles_type]
            || m_dynamic[gsd_flag::particles_image] || m_nframes == 0))
        {
        ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

        if (m_dynamic[gsd_flag::particles_position] || m_nframes == 0)
            {
            frame.particle_data_present[gsd_flag::particles_position] = true;
            }
        if (m_dynamic[gsd_flag::particles_image] || m_nframes == 0)
            {
            frame.particle_data_present[gsd_flag::particles_image] = true;
            }
        if (m_dynamic[gsd_flag::particles_type] || m_nframes == 0)
            {
            frame.particle_data_present[gsd_flag::particles_type] = true;
            }

        for (unsigned int index : m_index)
            {
            vec3<Scalar> position
                = vec3<Scalar>(h_postype.data[index]) - vec3<Scalar>(m_pdata->getOrigin());
            unsigned int type = __scalar_as_int(h_postype.data[index].w);
            int3 image = make_int3(0, 0, 0);

            if (m_dynamic[gsd_flag::particles_image] || m_nframes == 0)
                {
                image = h_image.data[index];
                }

            frame.global_box.wrap(position, image);

            if (m_dynamic[gsd_flag::particles_position] || m_nframes == 0)
                {
                if (position != vec3<Scalar>(0, 0, 0))
                    {
                    all_default[gsd_flag::particles_position] = false;
                    }

                frame.particle_data.pos.push_back(vec3<float>(position));
                }

            if (m_dynamic[gsd_flag::particles_image] || m_nframes == 0)
                {
                if (image != make_int3(0, 0, 0))
                    {
                    all_default[gsd_flag::particles_image] = false;
                    }

                frame.particle_data.image.push_back(image);
                }

            if (m_dynamic[gsd_flag::particles_type] || m_nframes == 0)
                {
                if (type != 0)
                    {
                    all_default[gsd_flag::particles_type] = false;
                    }

                frame.particle_data.type.push_back(type);
                }
            }
        }

    if (N > 0 && (m_dynamic[gsd_flag::particles_orientation] || m_nframes == 0))
        {
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        frame.particle_data_present[gsd_flag::particles_orientation] = true;

        for (unsigned int index : m_index)
            {
            quat<Scalar> orientation(h_orientation.data[index]);
            if (orientation.s != Scalar(1.0) || orientation.v.x != Scalar(0.0)
                || orientation.v.y != Scalar(0.0) || orientation.v.z != Scalar(0.0))
                {
                all_default[gsd_flag::particles_orientation] = false;
                }

            frame.particle_data.orientation.push_back(quat<float>(orientation));
            }
        }

    if (N > 0
        && (m_dynamic[gsd_flag::particles_velocity] || m_dynamic[gsd_flag::particles_mass]
            || m_nframes == 0))
        {
        ArrayHandle<Scalar4> h_velocity_mass(m_pdata->getVelocities(),
                                             access_location::host,
                                             access_mode::read);

        if (m_dynamic[gsd_flag::particles_mass] || m_nframes == 0)
            {
            frame.particle_data_present[gsd_flag::particles_mass] = true;
            }
        if (m_dynamic[gsd_flag::particles_velocity] || m_nframes == 0)
            {
            frame.particle_data_present[gsd_flag::particles_velocity] = true;
            }

        for (unsigned int index : m_index)
            {
            vec3<float> velocity = vec3<float>(static_cast<float>(h_velocity_mass.data[index].x),
                                               static_cast<float>(h_velocity_mass.data[index].y),
                                               static_cast<float>(h_velocity_mass.data[index].z));
            float mass = static_cast<float>(h_velocity_mass.data[index].w);

            if (m_dynamic[gsd_flag::particles_mass] || m_nframes == 0)
                {
                if (mass != 1.0f)
                    {
                    all_default[gsd_flag::particles_mass] = false;
                    }

                frame.particle_data.mass.push_back(mass);
                }

            if (m_dynamic[gsd_flag::particles_velocity] || m_nframes == 0)
                {
                if (velocity != vec3<float>(0, 0, 0))
                    {
                    all_default[gsd_flag::particles_velocity] = false;
                    }

                frame.particle_data.vel.push_back(velocity);
                }
            }
        }

    if (N > 0 && (m_dynamic[gsd_flag::particles_charge] || m_nframes == 0))
        {
        ArrayHandle<Scalar> h_charge(m_pdata->getCharges(),
                                     access_location::host,
                                     access_mode::read);

        frame.particle_data_present[gsd_flag::particles_charge] = true;

        for (unsigned int index : m_index)
            {
            float charge = static_cast<float>(h_charge.data[index]);
            if (charge != 0.0f)
                {
                all_default[gsd_flag::particles_charge] = false;
                }

            frame.particle_data.charge.push_back(charge);
            }
        }

    if (N > 0 && (m_dynamic[gsd_flag::particles_diameter] || m_nframes == 0))
        {
        ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(),
                                       access_location::host,
                                       access_mode::read);

        frame.particle_data_present[gsd_flag::particles_diameter] = true;

        for (unsigned int index : m_index)
            {
            float diameter = static_cast<float>(h_diameter.data[index]);

            if (diameter != 1.0f)
                {
                all_default[gsd_flag::particles_diameter] = false;
                }

            frame.particle_data.diameter.push_back(diameter);
            }
        }

    if (N > 0 && (m_dynamic[gsd_flag::particles_body] || m_nframes == 0))
        {
        ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                         access_location::host,
                                         access_mode::read);

        frame.particle_data_present[gsd_flag::particles_body] = true;

        for (unsigned int index : m_index)
            {
            unsigned int body = h_body.data[index];

            if (body != NO_BODY)
                {
                all_default[gsd_flag::particles_body] = false;
                }

            frame.particle_data.body.push_back(body);
            }
        }

    if (N > 0 && (m_dynamic[gsd_flag::particles_inertia] || m_nframes == 0))
        {
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::host,
                                       access_mode::read);

        frame.particle_data_present[gsd_flag::particles_inertia] = true;

        for (unsigned int index : m_index)
            {
            vec3<float> inertia = vec3<float>(h_inertia.data[index]);

            if (inertia != vec3<float>(0, 0, 0))
                {
                all_default[gsd_flag::particles_inertia] = false;
                }

            frame.particle_data.inertia.push_back(inertia);
            }
        }

    if (N > 0 && (m_dynamic[gsd_flag::particles_angmom] || m_nframes == 0))
        {
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::host,
                                      access_mode::read);

        frame.particle_data_present[gsd_flag::particles_angmom] = true;

        for (unsigned int index : m_index)
            {
            quat<float> angmom = quat<float>(h_angmom.data[index]);

            if (angmom.s != 0.0f || angmom.v.x != 0.0f || angmom.v.y != 0.0f || angmom.v.z != 0.0f)
                {
                all_default[gsd_flag::particles_angmom] = false;
                }

            frame.particle_data.angmom.push_back(angmom);
            }
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        unsigned long v = all_default.to_ulong();

        // All default only when all ranks are all default.
        MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_LONG, MPI_BAND, m_exec_conf->getMPICommunicator());

        all_default = std::bitset<n_gsd_flags>(v);

        // Present when any rank is present
        v = frame.particle_data_present.to_ulong();

        MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_LONG, MPI_BOR, m_exec_conf->getMPICommunicator());

        frame.particle_data_present = std::bitset<n_gsd_flags>(v);
        }
#endif

    // Keep data in arrays only when they are not all default or this is a non-zero frame
    // and the zeroth frame is non-default. To not keep, resize the arrays back to 0.
    // !(!all_default || (nframes > 0 && m_nondefault["value"])) <=>
    // (all_default && !(nframes > 0 && m_nondefault["value"])

    if (all_default[gsd_flag::particles_position]
        && !(m_nframes > 0 && m_nondefault["particles/position"]))
        {
        frame.particle_data.pos.resize(0);
        frame.particle_data_present[gsd_flag::particles_position] = false;
        }

    if (all_default[gsd_flag::particles_orientation]
        && !(m_nframes > 0 && m_nondefault["particles/orientation"]))
        {
        frame.particle_data.orientation.resize(0);
        frame.particle_data_present[gsd_flag::particles_orientation] = false;
        }

    if (all_default[gsd_flag::particles_type]
        && !(m_nframes > 0 && m_nondefault["particles/typeid"]))
        {
        frame.particle_data.type.resize(0);
        frame.particle_data_present[gsd_flag::particles_type] = false;
        }

    if (all_default[gsd_flag::particles_mass] && !(m_nframes > 0 && m_nondefault["particles/mass"]))
        {
        frame.particle_data.mass.resize(0);
        frame.particle_data_present[gsd_flag::particles_mass] = false;
        }

    if (all_default[gsd_flag::particles_charge]
        && !(m_nframes > 0 && m_nondefault["particles/charge"]))
        {
        frame.particle_data.charge.resize(0);
        frame.particle_data_present[gsd_flag::particles_charge] = false;
        }

    if (all_default[gsd_flag::particles_diameter]
        && !(m_nframes > 0 && m_nondefault["particles/diameter"]))
        {
        frame.particle_data.diameter.resize(0);
        frame.particle_data_present[gsd_flag::particles_diameter] = false;
        }

    if (all_default[gsd_flag::particles_body] && !(m_nframes > 0 && m_nondefault["particles/body"]))
        {
        frame.particle_data.body.resize(0);
        frame.particle_data_present[gsd_flag::particles_body] = false;
        }

    if (all_default[gsd_flag::particles_inertia]
        && !(m_nframes > 0 && m_nondefault["particles/moment_inertia"]))
        {
        frame.particle_data.inertia.resize(0);
        frame.particle_data_present[gsd_flag::particles_inertia] = false;
        }

    // momenta
    if (all_default[gsd_flag::particles_velocity]
        && !(m_nframes > 0 && m_nondefault["particles/velocity"]))
        {
        frame.particle_data.vel.resize(0);
        frame.particle_data_present[gsd_flag::particles_velocity] = false;
        }

    if (all_default[gsd_flag::particles_angmom]
        && !(m_nframes > 0 && m_nondefault["particles/angmom"]))
        {
        frame.particle_data.angmom.resize(0);
        frame.particle_data_present[gsd_flag::particles_angmom] = false;
        }

    if (all_default[gsd_flag::particles_image]
        && !(m_nframes > 0 && m_nondefault["particles/image"]))
        {
        frame.particle_data.image.resize(0);
        frame.particle_data_present[gsd_flag::particles_image] = false;
        }

    // capture topology data
    if (m_group->getNumMembersGlobal() != m_pdata->getNGlobal() && m_write_topology)
        {
        throw std::runtime_error("Cannot write topology for a portion of the system");
        }

    if (m_group->getNumMembersGlobal() == m_pdata->getNGlobal()
        && (m_write_topology || m_nframes == 0))
        {
        m_sysdef->getBondData()->takeSnapshot(frame.bond_data);
        m_sysdef->getAngleData()->takeSnapshot(frame.angle_data);
        m_sysdef->getDihedralData()->takeSnapshot(frame.dihedral_data);
        m_sysdef->getImproperData()->takeSnapshot(frame.improper_data);
        m_sysdef->getConstraintData()->takeSnapshot(frame.constraint_data);
        m_sysdef->getPairData()->takeSnapshot(frame.pair_data);
        }
    }

#ifdef ENABLE_MPI

/*! Gather per-particle data from the local frame and sort it into ascending tag order in
    m_global_frame.
*/
void GSDDumpWriter::gatherGlobalFrame(const GSDFrame& local_frame)
    {
    m_global_frame.clear();

    m_global_frame.timestep = local_frame.timestep;
    m_global_frame.global_box = local_frame.global_box;
    m_global_frame.particle_data.type_mapping = local_frame.particle_data.type_mapping;
    m_global_frame.particle_data_present = local_frame.particle_data_present;

    m_gather_tag_order.setLocalTagsSorted(local_frame.particle_tags);

    if (local_frame.particle_data_present[gsd_flag::particles_position])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.pos,
                                       local_frame.particle_data.pos);
        }

    if (local_frame.particle_data_present[gsd_flag::particles_orientation])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.orientation,
                                       local_frame.particle_data.orientation);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_type])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.type,
                                       local_frame.particle_data.type);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_mass])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.mass,
                                       local_frame.particle_data.mass);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_charge])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.charge,
                                       local_frame.particle_data.charge);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_diameter])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.diameter,
                                       local_frame.particle_data.diameter);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_body])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.body,
                                       local_frame.particle_data.body);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_inertia])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.inertia,
                                       local_frame.particle_data.inertia);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_velocity])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.vel,
                                       local_frame.particle_data.vel);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_angmom])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.angmom,
                                       local_frame.particle_data.angmom);
        }
    if (local_frame.particle_data_present[gsd_flag::particles_image])
        {
        m_gather_tag_order.gatherArray(m_global_frame.particle_data.image,
                                       local_frame.particle_data.image);
        }
    }

#endif

namespace detail
    {
void export_GSDDumpWriter(pybind11::module& m)
    {
    pybind11::bind_map<std::map<std::string, pybind11::function>>(m, "MapStringFunction");

    pybind11::class_<GSDDumpWriter, Analyzer, std::shared_ptr<GSDDumpWriter>>(m, "GSDDumpWriter")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::string,
                            std::shared_ptr<ParticleGroup>,
                            std::string,
                            bool>())
        .def_property("log_writer", &GSDDumpWriter::getLogWriter, &GSDDumpWriter::setLogWriter)
        .def_property_readonly("filename", &GSDDumpWriter::getFilename)
        .def_property_readonly("mode", &GSDDumpWriter::getMode)
        .def_property("dynamic", &GSDDumpWriter::getDynamic, &GSDDumpWriter::setDynamic)
        .def_property_readonly("truncate", &GSDDumpWriter::getTruncate)
        .def_property_readonly("filter",
                               [](const std::shared_ptr<GSDDumpWriter> gsd)
                               { return gsd->getGroup()->getFilter(); })
        .def_property("write_diameter",
                      &GSDDumpWriter::getWriteDiameter,
                      &GSDDumpWriter::setWriteDiameter)
        .def("flush", &GSDDumpWriter::flush)
        .def_property("maximum_write_buffer_size",
                      &GSDDumpWriter::getMaximumWriteBufferSize,
                      &GSDDumpWriter::setMaximumWriteBufferSize);
    }

    } // end namespace detail

    } // end namespace hoomd
