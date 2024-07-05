// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file DCDDumpWriter.cc
    \brief Defines the DCDDumpWriter class and related helper functions
*/

#include "DCDDumpWriter.h"
#include "Filesystem.h"
#include "time.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <limits>
#include <stdexcept>

using namespace std;

// File position of NFILE in DCD header
#define NFILE_POS 8L
// File position of NSTEP in DCD header
#define NSTEP_POS 20L

namespace hoomd
    {
namespace detail
    {
//! simple helper function to write an integer
/*! \param file file to write to
    \param val integer to write
*/
static void write_int(fstream& file, unsigned int val)
    {
    file.write((char*)&val, sizeof(unsigned int));
    }

//! simple helper function to read in integer
/*! \param file file to read from
    \returns integer read
*/
static unsigned int read_int(fstream& file)
    {
    unsigned int val;
    file.read((char*)&val, sizeof(unsigned int));
    return val;
    }
    } // end namespace detail

/*! Constructs the DCDDumpWriter. After construction, settings are set. No file operations are
    attempted until analyze() is called.

    \param sysdef SystemDefinition containing the ParticleData to dump
    \param fname File name to write DCD data to
    \param period Period which analyze() is going to be called at
    \param group Group of particles to include in the output
    \param overwrite If false, existing files will be appended to. If true, existing files will be
   overwritten.

    \note You must call analyze() with the same period specified in the constructor or
    the time step information in the file will be invalid. analyze() will print a warning
    if it is called out of sequence.
*/
DCDDumpWriter::DCDDumpWriter(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<Trigger> trigger,
                             const std::string& fname,
                             unsigned int period,
                             std::shared_ptr<ParticleGroup> group,
                             bool overwrite)
    : Analyzer(sysdef, trigger), m_fname(fname), m_start_timestep(0), m_period(period),
      m_group(group), m_num_frames_written(0), m_last_written_step(0), m_appending(false),
      m_unwrap_full(false), m_unwrap_rigid(false), m_angle(false), m_overwrite(overwrite),
      m_is_initialized(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing DCDDumpWriter: " << fname << " " << period << " "
                                << overwrite << endl;
    }

//! Initializes the output file for writing
void DCDDumpWriter::initFileIO(uint64_t timestep)
    {
    m_staging_buffer = new float[m_pdata->getNGlobal()];
    m_is_initialized = true;

    m_nglobal = m_pdata->getNGlobal();

    // handle appending to an existing file if it is requested
    if (!m_overwrite && filesystem::exists(m_fname))
        {
        m_exec_conf->msg->notice(3)
            << "DCD: Appending to existing DCD file \"" << m_fname << "\"" << endl;

        // open the file and get data from the header
        m_file.open(m_fname.c_str(), ios::in | ios::out | ios::binary);
        m_file.seekp(NFILE_POS);

        m_num_frames_written = detail::read_int(m_file);
        m_start_timestep = detail::read_int(m_file);
        unsigned int file_period = detail::read_int(m_file);

        m_exec_conf->msg->notice(4) << "DCD: File has " << m_num_frames_written
                                    << " frames | start timestep: " << m_start_timestep
                                    << " | and period " << m_period << endl;

        // warn the user if we are now dumping at a different period
        if (file_period != m_period)
            m_exec_conf->msg->warning()
                << "DCD: appending to a file that has period " << file_period
                << " that is not the same as the requested period of " << m_period << endl;

        m_last_written_step = detail::read_int(m_file);

        // check for errors
        if (!m_file.good())
            {
            throw runtime_error("DCD: I/O error while reading DCD header data");
            }

        m_appending = true;
        }
    else
        {
        // open the file and truncate it
        m_file.open(m_fname.c_str(), ios::trunc | ios::out | ios::binary);

        // write the file header
        m_start_timestep = timestep;
        write_file_header(m_file);
        }
    }

DCDDumpWriter::~DCDDumpWriter()
    {
    m_exec_conf->msg->notice(5) << "Destroying DCDDumpWriter" << endl;

    if (m_is_initialized)
        {
        m_file.close();
        delete[] m_staging_buffer;
        }
    }

/*! \param timestep Current time step of the simulation
    The very first call to analyze() will result in the creation (or overwriting) of the
    file fname and the writing of the current timestep snapshot. After that, each call to analyze
    will add a new snapshot to the end of the file.
*/
void DCDDumpWriter::analyze(uint64_t timestep)
    {
    Analyzer::analyze(timestep);
    // take particle data snapshot
    SnapshotParticleData<Scalar> snapshot;

    m_pdata->takeSnapshot(snapshot);

#ifdef ENABLE_MPI
    // if we are not the root processor, do not perform file I/O
    if (m_sysdef->isDomainDecomposed() && !m_exec_conf->isRoot())
        {
        return;
        }
#endif

    if (!m_is_initialized)
        initFileIO(timestep);

    if (m_nglobal != m_pdata->getNGlobal())
        {
        m_exec_conf->msg->error()
            << "analyze.dcd: Change in number of particles unsupported by DCD file format."
            << std::endl;
        throw std::runtime_error("Error writing DCD file");
        }

    if (m_appending && timestep <= m_last_written_step)
        {
        m_exec_conf->msg->warning()
            << "DCD: not writing output at timestep " << timestep
            << " because the file reports that it already has data up to step "
            << m_last_written_step << endl;
        return;
        }

    // verify the period on subsequent frames
    if ((timestep - m_start_timestep) % m_period != 0)
        m_exec_conf->msg->warning()
            << "DCD: writing time step " << timestep
            << " which is not specified in the period of the DCD file: " << m_start_timestep
            << " + i * " << m_period << endl;

    // write the data for the current time step
    m_file.seekp(0, std::ios_base::end);
    write_frame_header(m_file);
    write_frame_data(m_file, snapshot);

    // update the header with the number of frames written
    m_num_frames_written++;
    write_updated_header(m_file, timestep);
    }

/*! \param file File to write to
    Writes the initial DCD header to the beginning of the file. This must be
    called on a newly created (or truncated file).
*/
void DCDDumpWriter::write_file_header(std::fstream& file)
    {
    m_exec_conf->msg->notice(4) << "DCD: Creating dcd file "
                                << " | start timestep: " << m_start_timestep << " | and period "
                                << m_period << endl;

    if (m_start_timestep > std::numeric_limits<uint32_t>::max())
        m_exec_conf->msg->warning() << "DCD: Truncating timestep to lower 32 bits" << endl;

    // the first 4 bytes in the file must be 84
    detail::write_int(file, 84);

    // the next 4 bytes in the file must be "CORD"
    char cord_data[] = "CORD";
    file.write(cord_data, 4);
    detail::write_int(file, 0); // Number of frames in file, none written yet
    detail::write_int(file, static_cast<uint32_t>(m_start_timestep)); // Starting timestep
    detail::write_int(file, m_period); // Timesteps between frames written to the file
    detail::write_int(file, 0);        // Number of timesteps in simulation
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0); // timestep (unused)
    detail::write_int(file, 1); // include unit cell
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 0);
    detail::write_int(file, 24); // Pretend to be CHARMM version 24
    detail::write_int(file, 84);
    detail::write_int(file, 164);
    detail::write_int(file, 2);

    char title_string[81];
    memset(title_string, 0, 81);
    char remarks[] = "Created by HOOMD";
    strncpy(title_string, remarks, 80);
    title_string[79] = '\0';
    file.write(title_string, 80);

    char time_str[81];
    memset(time_str, 0, 81);
    time_t cur_time = time(NULL);
    tm* tmbuf = localtime(&cur_time);
    strftime(time_str, 80, "REMARKS Created  %d %B, %Y at %H:%M", tmbuf);
    file.write(time_str, 80);

    detail::write_int(file, 164);
    detail::write_int(file, 4);
    unsigned int nparticles = m_group->getNumMembersGlobal();
    detail::write_int(file, nparticles);
    detail::write_int(file, 4);

    // check for errors
    if (!file.good())
        {
        throw runtime_error("DCD: I/O error when writing DCD header.");
        }
    }

/*! \param file File to write to
    Writes the header that precedes each snapshot in the file. This header
    includes information on the box size of the simulation.
*/
void DCDDumpWriter::write_frame_header(std::fstream& file)
    {
    double unitcell[6];
    BoxDim box = m_pdata->getGlobalBox();
    // set box dimensions
    Scalar a, b, c, alpha, beta, gamma;
    Scalar3 va = box.getLatticeVector(0);
    Scalar3 vb = box.getLatticeVector(1);
    Scalar3 vc = box.getLatticeVector(2);
    a = sqrt(dot(va, va));
    b = sqrt(dot(vb, vb));
    c = sqrt(dot(vc, vc));
    alpha = dot(vb, vc) / (b * c);
    beta = dot(va, vc) / (a * c);
    gamma = dot(va, vb) / (a * b);

    unitcell[0] = a;
    unitcell[2] = b;
    unitcell[5] = c;
    // box angles are 90 degrees
    unitcell[1] = gamma;
    unitcell[3] = beta;
    unitcell[4] = alpha;

    detail::write_int(file, 48);
    file.write((char*)unitcell, 48);
    detail::write_int(file, 48);

    // check for errors
    if (!file.good())
        {
        throw runtime_error("DCD: I/O error while writing DCD frame header.");
        }
    }

/*! \param file File to write to
    \param snapshot Snapshot to write
    Writes the actual particle positions for all particles at the current time step
*/
void DCDDumpWriter::write_frame_data(std::fstream& file,
                                     const SnapshotParticleData<Scalar>& snapshot)
    {
    // we need to unsort the positions and write in tag order
    assert(m_staging_buffer);

    BoxDim box = m_pdata->getGlobalBox();

    unsigned int nparticles = m_group->getNumMembersGlobal();

    // Create a tmp copy of the particle data and unwrap particles
    std::vector<vec3<Scalar>> tmp_pos(snapshot.pos);
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int i = m_group->getMemberTag(group_idx);

        if (m_unwrap_full)
            {
            tmp_pos[i] = box.shift(tmp_pos[i], snapshot.image[i]);
            }
        else if (m_unwrap_rigid && snapshot.body[i] < MIN_FLOPPY)
            {
            unsigned int central_ptl_tag = snapshot.body[i];
            int body_ix = snapshot.image[central_ptl_tag].x;
            int body_iy = snapshot.image[central_ptl_tag].y;
            int body_iz = snapshot.image[central_ptl_tag].z;
            int3 particle_img = snapshot.image[i];
            int3 img_diff = make_int3(particle_img.x - body_ix,
                                      particle_img.y - body_iy,
                                      particle_img.z - body_iz);

            tmp_pos[i] = box.shift(tmp_pos[i], img_diff);
            }
        }

    // prepare x coords for writing, looping in tag order
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int i = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = float(tmp_pos[i].x);
        }

    // write x coords
    detail::write_int(file, (unsigned int)(nparticles * sizeof(float)));
    file.write((char*)m_staging_buffer, nparticles * sizeof(float));
    detail::write_int(file, (unsigned int)(nparticles * sizeof(float)));

    // prepare y coords for writing
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int i = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = float(tmp_pos[i].y);
        }

    // write y coords
    detail::write_int(file, (unsigned int)(nparticles * sizeof(float)));
    file.write((char*)m_staging_buffer, nparticles * sizeof(float));
    detail::write_int(file, (unsigned int)(nparticles * sizeof(float)));

    // prepare z coords for writing
    for (unsigned int group_idx = 0; group_idx < nparticles; group_idx++)
        {
        unsigned int i = m_group->getMemberTag(group_idx);
        m_staging_buffer[group_idx] = float(tmp_pos[i].z);

        // m_angle set to True turns on a hack where the particle orientation angle is written out
        // to the z component this only works in 2D simulations, obviously
        if (m_angle)
            {
            m_staging_buffer[group_idx]
                = float(atan2(snapshot.orientation[i].v.z, snapshot.orientation[i].s) * 2);
            }
        }

    // write z coords
    detail::write_int(file, (unsigned int)(nparticles * sizeof(float)));
    file.write((char*)m_staging_buffer, nparticles * sizeof(float));
    detail::write_int(file, (unsigned int)(nparticles * sizeof(float)));

    // check for errors
    if (!file.good())
        {
        throw runtime_error("I/O error while writing DCD frame data.");
        }
    }

/*! \param file File to write to
    \param timestep Current time step of the simulation

    Updates the pointers in the main file header to reflect the current number of frames
    written and the last time step written.
*/
void DCDDumpWriter::write_updated_header(std::fstream& file, uint64_t timestep)
    {
    file.seekp(NFILE_POS);
    detail::write_int(file, m_num_frames_written);

    file.seekp(NSTEP_POS);
    detail::write_int(file, static_cast<uint32_t>(timestep));

    if (timestep > std::numeric_limits<uint32_t>::max())
        m_exec_conf->msg->warning() << "DCD: Truncating timestep to lower 32 bits" << endl;
    }

namespace detail
    {
void export_DCDDumpWriter(pybind11::module& m)
    {
    pybind11::class_<DCDDumpWriter, Analyzer, std::shared_ptr<DCDDumpWriter>>(m, "DCDDumpWriter")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::string,
                            unsigned int,
                            std::shared_ptr<ParticleGroup>,
                            bool>())
        .def_property("unwrap_full", &DCDDumpWriter::getUnwrapFull, &DCDDumpWriter::setUnwrapFull)
        .def_property("unwrap_rigid",
                      &DCDDumpWriter::getUnwrapRigid,
                      &DCDDumpWriter::setUnwrapRigid)
        .def_property("angle_z", &DCDDumpWriter::getAngleZ, &DCDDumpWriter::setAngleZ)
        .def_property_readonly("overwrite", &DCDDumpWriter::getOverwrite);
    }
    } // end namespace detail

    } // end namespace hoomd
