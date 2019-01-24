// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file MSDAnalyzer.cc
    \brief Defines the MSDAnalyzer class
*/



#include "MSDAnalyzer.h"
#include "HOOMDInitializer.h"
#include "hoomd/Filesystem.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace py = pybind11;

#include <iomanip>
using namespace std;

/*! \param sysdef SystemDefinition containing the Particle data to analyze
    \param fname File name to write output to
    \param header_prefix String to print before the file header
    \param overwrite Will overwrite an exiting file if true (default is to append)

    On construction, the initial coordinates of all particles in the system are recorded. The file is opened
    (and overwritten if told to). Nothing is initially written to the file, that will occur on the first call to
    analyze()
*/
MSDAnalyzer::MSDAnalyzer(std::shared_ptr<SystemDefinition> sysdef,
                         std::string fname,
                         const std::string& header_prefix,
                         bool overwrite)
    : Analyzer(sysdef), m_delimiter("\t"), m_header_prefix(header_prefix), m_appending(false),
      m_columns_changed(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing MSDAnalyzer: " << fname << " " << header_prefix << " " << overwrite << endl;

    SnapshotParticleData<Scalar> snapshot;

    m_pdata->takeSnapshot(snapshot);

#ifdef ENABLE_MPI
    // if we are not the root processor, do not perform file I/O
    if (m_comm && !m_exec_conf->isRoot())
        {
        return;
        }
#endif

    // open the file
    if (filesystem::exists(fname) && !overwrite)
        {
        m_exec_conf->msg->notice(3) << "analyze.msd: Appending msd to existing file \"" << fname << "\"" << endl;
        m_file.open(fname.c_str(), ios_base::in | ios_base::out | ios_base::ate);
        m_appending = true;
        }
    else
        {
        m_exec_conf->msg->notice(3) << "analyze.msd: Creating new msd in file \"" << fname << "\"" << endl;
        m_file.open(fname.c_str(), ios_base::out);
        }

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "analyze.msd: Unable to open file " << fname << endl;
        throw runtime_error("Error initializing analyze.msd");
        }

    // record the initial particle positions by tag
    m_initial_x.resize(m_pdata->getNGlobal());
    m_initial_y.resize(m_pdata->getNGlobal());
    m_initial_z.resize(m_pdata->getNGlobal());

    BoxDim box = m_pdata->getGlobalBox();

    // for each particle in the data
    for (unsigned int tag = 0; tag < snapshot.size; tag++)
        {
        // save its initial position
        vec3<Scalar> pos = snapshot.pos[tag];
        vec3<Scalar> unwrapped = box.shift(pos, snapshot.image[tag]);
        m_initial_x[tag] = unwrapped.x;
        m_initial_y[tag] = unwrapped.y;
        m_initial_z[tag] = unwrapped.z;
        }

    m_pdata->getParticleSortSignal().connect<MSDAnalyzer, &MSDAnalyzer::slotParticleSort>(this);
    }

MSDAnalyzer::~MSDAnalyzer()
    {
    m_exec_conf->msg->notice(5) << "Destroying MSDAnalyzer" << endl;

    m_pdata->getParticleSortSignal().disconnect<MSDAnalyzer, &MSDAnalyzer::slotParticleSort>(this);
    }

void MSDAnalyzer::slotParticleSort()
    {
    // check if any of the groups has changed size
    for (unsigned int i = 0; i < m_columns.size(); i++)
        {
        if (m_columns[i].m_group->getNumMembersGlobal() != m_initial_group_N[i])
            {
            m_exec_conf->msg->error() << "analyze.msd: Change in number of particles for column " << m_columns[i].m_name << " unsupported." << std::endl;
            throw std::runtime_error("Error adding/removing particles");
            }
        }
    }

/*!\param timestep Current time step of the simulation

    analyze() will first write out the file header if the columns have changed.

    On every call, analyze() will write calculate the MSD for each group and write out a row in the file.
*/
void MSDAnalyzer::analyze(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Analyze MSD");

    // take particle data snapshot
    SnapshotParticleData<Scalar> snapshot(0);

    m_pdata->takeSnapshot(snapshot);

#ifdef ENABLE_MPI
    // if we are not the root processor, do not perform file I/O
    if (m_comm && !m_exec_conf->isRoot())
        {
        if (m_prof) m_prof->pop();
        return;
        }
#endif

    // error check
    if (m_columns.size() == 0)
        {
        m_exec_conf->msg->warning() << "analyze.msd: No columns specified in the MSD analysis" << endl;
        return;
        }

    // ignore writing the header on the first call when appending the file
    if (m_columns_changed && m_appending)
        {
        m_appending = false;
        m_columns_changed = false;
        }

    // write out the header only once if the columns change
    if (m_columns_changed)
        {
        writeHeader();
        m_columns_changed = false;
        }

    // write out the row every time
    writeRow(timestep, snapshot);

    if (m_prof)
        m_prof->pop();
    }

/*! \param delimiter New delimiter to set

    The delimiter is printed between every element in the row of the output
*/
void MSDAnalyzer::setDelimiter(const std::string& delimiter)
    {
    m_delimiter = delimiter;
    }

/*! \param group Particle group to calculate the MSD of
    \param name Name to print in the header of the file

    After a column is added with addColumn(), future calls to analyze() will calculate the MSD of the particles defined
    in \a group and print out an entry under the \a name header in the file.
*/
void MSDAnalyzer::addColumn(std::shared_ptr<ParticleGroup> group, const std::string& name)
    {
    m_columns.push_back(column(group, name));

    // store initial number of particles
    m_initial_group_N.push_back(group->getNumMembersGlobal());

    m_columns_changed = true;
    }

/*! \param xml_fname Name of the XML file to read in to the r0 positions

    \post \a xml_fname is read and all initial r0 positions are assigned from that file.
*/
void MSDAnalyzer::setR0(const std::string& xml_fname)
    {
    // read in the xml file
    HOOMDInitializer xml(m_exec_conf,xml_fname);

#ifdef ENABLE_MPI
    // if we are not the root processor, do not perform file I/O
    if (m_pdata->getDomainDecomposition() && !m_exec_conf->isRoot())
        {
        return;
        }
#endif

    // verify that the input matches the current system size
    unsigned int nparticles = m_pdata->getNGlobal();
    if (nparticles != xml.getPos().size())
        {
        m_exec_conf->msg->error() << "analyze.msd: Found " << xml.getPos().size() << " particles in "
             << xml_fname << ", but there are " << nparticles << " in the current simulation." << endl;
        throw runtime_error("Error setting r0 in analyze.msd");
        }

    // determine if we have image data
    bool have_image = (xml.getImage().size() == nparticles);
    if (!have_image)
        {
        m_exec_conf->msg->warning() << "analyze.msd: Image data missing or corrupt in " << xml_fname
             << ". Computed msd values will not be correct." << endl;
        }

    // reset the initial positions
    BoxDim box = m_pdata->getGlobalBox();

    // for each particle in the data
    for (unsigned int tag = 0; tag < nparticles; tag++)
        {
        // save its initial position
        HOOMDInitializer::vec pos = xml.getPos()[tag];
        m_initial_x[tag] = pos.x;
        m_initial_y[tag] = pos.y;
        m_initial_z[tag] = pos.z;

        // adjust the positions by the image flags if we have them
        if (have_image)
            {
            HOOMDInitializer::vec_int image = xml.getImage()[tag];
            Scalar3 pos = make_scalar3(m_initial_x[tag], m_initial_y[tag], m_initial_z[tag]);
            int3 image_i = make_int3(image.x, image.y, image.z);
            Scalar3 unwrapped = box.shift(pos, image_i);
            m_initial_x[tag] = unwrapped.x;
            m_initial_y[tag] = unwrapped.y;
            m_initial_z[tag] = unwrapped.z;
            }
        }
    }

/*! The entire header row is written to the file. First, timestep is written as every file includes it and then the
    columns are looped through and their names printed, separated by the delimiter.
*/
void MSDAnalyzer::writeHeader()
    {
    // write out the header prefix
    m_file << m_header_prefix;

    // timestep is always output
    m_file << "timestep";

    if (m_columns.size() == 0)
        {
        m_exec_conf->msg->warning() << "analyze.msd: No columns specified in the MSD analysis" << endl;
        return;
        }

    // only print the delimiter after the timestep if there are more columns
    m_file << m_delimiter;

    // write all but the last of the quantities separated by the delimiter
    for (unsigned int i = 0; i < m_columns.size()-1; i++)
        m_file << m_columns[i].m_name << m_delimiter;
    // write the last one with no delimiter after it
    m_file << m_columns[m_columns.size()-1].m_name << endl;
    m_file.flush();
    }

/*! \param group Particle group to calculate the MSD of
    Loop through all particles in the given group and calculate the MSD over them.
    \returns The calculated MSD
*/
Scalar MSDAnalyzer::calcMSD(std::shared_ptr<ParticleGroup const> group, const SnapshotParticleData<Scalar>& snapshot)
    {
    BoxDim box = m_pdata->getGlobalBox();

    // initial sum for the average
    Scalar msd = Scalar(0.0);

    // handle the case where there are 0 members gracefully
    if (group->getNumMembersGlobal() == 0)
        {
        m_exec_conf->msg->warning() << "analyze.msd: Group has 0 members, reporting a calculated msd of 0.0" << endl;
        return Scalar(0.0);
        }

    // for each particle in the group
    for (unsigned int group_idx = 0; group_idx < group->getNumMembersGlobal(); group_idx++)
        {
        // get the tag for the current group member from the group
        unsigned int tag = group->getMemberTag(group_idx);
        assert(tag < snapshot.size);
        vec3<Scalar> pos = snapshot.pos[tag];
        int3 image = snapshot.image[tag];
        vec3<Scalar> unwrapped = box.shift(pos, image);
        Scalar dx = unwrapped.x - m_initial_x[tag];
        Scalar dy = unwrapped.y - m_initial_y[tag];
        Scalar dz = unwrapped.z - m_initial_z[tag];

        msd += dx*dx + dy*dy + dz*dz;
        }

    // divide to complete the average
    msd /= Scalar(group->getNumMembersGlobal());
    return msd;
    }

/*! \param timestep current time step of the simulation

    Performs all the steps needed in order to calculate the MSDs for all the groups in the columns and writes out an
    entire row to the file.
*/
void MSDAnalyzer::writeRow(unsigned int timestep, const SnapshotParticleData<Scalar>& snapshot)
    {
    if (m_prof) m_prof->push("MSD");

    // The timestep is always output
    m_file << setprecision(10) << timestep;

    // quit now if there is nothing to log
    if (m_columns.size() == 0)
        {
        return;
        }

    // only print the delimiter after the timestep if there are more columns
    m_file << m_delimiter;

    // write all but the last of the columns separated by the delimiter
    for (unsigned int i = 0; i < m_columns.size()-1; i++)
        m_file << setprecision(10) << calcMSD(m_columns[i].m_group, snapshot) << m_delimiter;
    // write the last one with no delimiter after it
    m_file << setprecision(10) << calcMSD(m_columns[m_columns.size()-1].m_group, snapshot) << endl;
    m_file.flush();

    if (!m_file.good())
        {
        m_exec_conf->msg->error() << "analyze.msd: I/O error while writing file" << endl;
        throw runtime_error("Error writing msd file");
        }

    if (m_prof) m_prof->pop();
    }

void export_MSDAnalyzer(py::module& m)
    {
    py::class_<MSDAnalyzer, std::shared_ptr<MSDAnalyzer> >(m,"MSDAnalyzer",py::base<Analyzer>())
    .def(py::init< std::shared_ptr<SystemDefinition>, const std::string&, const std::string&, bool >())
    .def("setDelimiter", &MSDAnalyzer::setDelimiter)
    .def("addColumn", &MSDAnalyzer::addColumn)
    .def("setR0", &MSDAnalyzer::setR0)
    ;
    }
