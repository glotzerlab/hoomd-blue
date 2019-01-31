// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

/*! \file HOOMDDumpWriter.cc
    \brief Defines the HOOMDDumpWriter class
*/

#include "HOOMDDumpWriter.h"
#include "hoomd/BondedGroupData.h"

namespace py = pybind11;

#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <memory>

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

using namespace std;

/*! \param sysdef SystemDefinition containing the ParticleData to dump
    \param base_fname The base name of the file xml file to output the information
    \param group ParticleGroup to dump to file
    \param mode_restart Set to true to enable restart writing mode. False writes one XML file per time step.

    \note .timestep.xml will be appended to the end of \a base_fname when analyze() is called.
*/
HOOMDDumpWriter::HOOMDDumpWriter(std::shared_ptr<SystemDefinition> sysdef,
                                 const std::string& base_fname,
                                 std::shared_ptr<ParticleGroup> group,
                                 bool mode_restart)
        : Analyzer(sysdef), m_base_fname(base_fname), m_group(group), m_output_position(true),
        m_output_image(false), m_output_velocity(false), m_output_mass(false), m_output_diameter(false),
        m_output_type(false), m_output_bond(false), m_output_angle(false),
        m_output_dihedral(false), m_output_improper(false), m_output_constraint(false),
        m_output_accel(false), m_output_body(false),
        m_output_charge(false), m_output_orientation(false), m_output_angmom(false),
        m_output_moment_inertia(false), m_vizsigma_set(false), m_mode_restart(mode_restart)
    {
    m_exec_conf->msg->notice(5) << "Constructing HOOMDDumpWriter: " << base_fname << endl;
    }

HOOMDDumpWriter::~HOOMDDumpWriter()
    {
    m_exec_conf->msg->notice(5) << "Destroying HOOMDDumpWriter" << endl;
    }

/*! \param enable Set to true to enable the writing of particle positions to the files in analyze()
*/
void HOOMDDumpWriter::setOutputPosition(bool enable)
    {
    m_output_position = enable;
    }

/*! \param enable Set to true to enable the writing of particle images to the files in analyze()
*/
void HOOMDDumpWriter::setOutputImage(bool enable)
    {
    m_output_image = enable;
    }

/*!\param enable Set to true to output particle velocities to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputVelocity(bool enable)
    {
    m_output_velocity = enable;
    }

/*!\param enable Set to true to output particle masses to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputMass(bool enable)
    {
    m_output_mass = enable;
    }

/*!\param enable Set to true to output particle diameters to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputDiameter(bool enable)
    {
    m_output_diameter = enable;
    }

/*! \param enable Set to true to output particle types to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputType(bool enable)
    {
    m_output_type = enable;
    }
/*! \param enable Set to true to output bonds to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputBond(bool enable)
    {
    m_output_bond = enable;
    }
/*! \param enable Set to true to output angles to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputAngle(bool enable)
    {
    m_output_angle = enable;
    }
/*! \param enable Set to true to output dihedrals to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputDihedral(bool enable)
    {
    m_output_dihedral = enable;
    }
/*! \param enable Set to true to output impropers to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputImproper(bool enable)
    {
    m_output_improper = enable;
    }

/*! \param enable Set to true to output constraints to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputConstraint(bool enable)
    {
    m_output_constraint = enable;
    }

/*! \param enable Set to true to output acceleration to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputAccel(bool enable)
    {
    m_output_accel = enable;
    }
/*! \param enable Set to true to output body to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputBody(bool enable)
    {
    m_output_body = enable;
    }

/*! \param enable Set to true to output body to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputCharge(bool enable)
    {
    m_output_charge = enable;
    }

/*! \param enable Set to true to output orientation to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputOrientation(bool enable)
    {
    m_output_orientation = enable;
    }

/*! \param enable Set to true to output orientation to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputAngularMomentum(bool enable)
    {
    m_output_angmom = enable;
    }

/*! \param enable Set to true to output moment_inertia to the XML file on the next call to analyze()
*/
void HOOMDDumpWriter::setOutputMomentInertia(bool enable)
    {
    m_output_moment_inertia = enable;
    }

/*! \param fname File name to write
    \param timestep Current time step of the simulation
*/
void HOOMDDumpWriter::writeFile(std::string fname, unsigned int timestep)
    {
    // acquire the particle data
    SnapshotParticleData<Scalar> snapshot;

    const auto map = m_pdata->takeSnapshot(snapshot);

    BondData::Snapshot bdata_snapshot;
    if (m_output_bond) m_sysdef->getBondData()->takeSnapshot(bdata_snapshot);

    AngleData::Snapshot adata_snapshot;
    if (m_output_angle) m_sysdef->getAngleData()->takeSnapshot(adata_snapshot);

    DihedralData::Snapshot ddata_snapshot;
    if (m_output_dihedral) m_sysdef->getDihedralData()->takeSnapshot(ddata_snapshot);

    ImproperData::Snapshot idata_snapshot;
    if (m_output_improper) m_sysdef->getImproperData()->takeSnapshot(idata_snapshot);

    ConstraintData::Snapshot cdata_snapshot;
    if (m_output_constraint) m_sysdef->getConstraintData()->takeSnapshot(cdata_snapshot);


#ifdef ENABLE_MPI
    // only the root processor writes the output file
    if (m_pdata->getDomainDecomposition() && ! m_exec_conf->isRoot())
        return;
#endif

    // open the file for writing
    ofstream f(fname.c_str());

    if (!f.good())
        {
        m_exec_conf->msg->error() << "dump.xml: Unable to open dump file for writing: " << fname << endl;
        throw runtime_error("Error writing hoomd_xml dump file");
        }

    BoxDim box = m_pdata->getGlobalBox();
    Scalar3 L = box.getL();
    Scalar xy = box.getTiltFactorXY();
    Scalar xz = box.getTiltFactorXZ();
    Scalar yz = box.getTiltFactorYZ();

    // number of members in the group
    const unsigned int N = m_group->getNumMembersGlobal();

    f.precision(13);
    f << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << "\n";
    f << "<hoomd_xml version=\"1.7\">" << "\n";
    f << "<configuration time_step=\"" << timestep << "\" "
      << "dimensions=\"" << m_sysdef->getNDimensions() << "\" "
      << "natoms=\"" << N << "\" ";
    if (m_vizsigma_set)
        f << "vizsigma=\"" << m_vizsigma << "\" ";
    f << ">" << "\n";
    f << "<box " << "lx=\"" << L.x << "\" ly=\""<< L.y << "\" lz=\""<< L.z
      << "\" xy=\"" << xy << "\" xz=\"" << xz << "\" yz=\"" << yz << "\"/>" << "\n";

    f.precision(12);

    // If the position flag is true output the position of all particles to the file
    if (m_output_position)
        {
        f << "<position num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            vec3<Scalar> pos = snapshot.pos[it->second];

            f << pos.x << " " << pos.y << " "<< pos.z << "\n";

            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</position>" << "\n";
        }

    // If the image flag is true, output the image of each particle to the file
    if (m_output_image)
        {
        f << "<image num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            int3 image = snapshot.image[it->second];

            f << image.x << " " << image.y << " "<< image.z << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</image>" << "\n";
        }

    // If the velocity flag is true output the velocity of all particles to the file
    if (m_output_velocity)
        {
        f <<"<velocity num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            vec3<Scalar> vel = snapshot.vel[it->second];

            f << vel.x << " " << vel.y << " " << vel.z << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</velocity>" << "\n";
        }

    // If the velocity flag is true output the velocity of all particles to the file
    if (m_output_accel)
        {
        f <<"<acceleration num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            vec3<Scalar> accel = snapshot.accel[it->second];

            f << accel.x << " " << accel.y << " " << accel.z << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</acceleration>" << "\n";
        }

    // If the mass flag is true output the mass of all particles to the file
    if (m_output_mass)
        {
        f <<"<mass num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            Scalar mass = snapshot.mass[it->second];

            f << mass << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</mass>" << "\n";
        }

    // If the charge flag is true output the mass of all particles to the file
    if (m_output_charge)
        {
        f <<"<charge num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            Scalar charge = snapshot.charge[it->second];

            f << charge << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</charge>" << "\n";
        }

    // If the diameter flag is true output the mass of all particles to the file
    if (m_output_diameter)
        {
        f <<"<diameter num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            Scalar diameter = snapshot.diameter[it->second];

            f << diameter << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</diameter>" << "\n";
        }

    // If the Type flag is true output the types of all particles to an xml file
    if  (m_output_type)
        {
        f <<"<type num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            unsigned int type = snapshot.type[it->second];

            f << m_pdata->getNameByType(type) << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</type>" << "\n";
        }

    // If the body flag is true output the bodies of all particles to an xml file
    if  (m_output_body)
        {
        f <<"<body num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            unsigned int body = snapshot.body[it->second];
            int out = (body == NO_BODY) ? -1 : (int)body;

            f << out << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f <<"</body>" << "\n";
        }

    // if the orientation flag is set, write out the orientation quaternion to the XML file
    if (m_output_orientation)
        {
        f << "<orientation num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            Scalar4 orientation = quat_to_scalar4(snapshot.orientation[it->second]);

            f << orientation.x << " " << orientation.y << " " << orientation.z << " " << orientation.w << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }
        f << "</orientation>" << "\n";
        }

    // if the angmom flag is set, write out the angular momentum quaternion to the XML file
    if (m_output_angmom)
        {
        f << "<angmom num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            Scalar4 angmom = quat_to_scalar4(snapshot.angmom[it->second]);

            f << angmom.x << " " << angmom.y << " " << angmom.z << " " << angmom.w << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f << "</angmom>" << "\n";
        }

    // if the moment_inertia flag is set, write out the principal moments of inertia to the XML file
    if (m_output_moment_inertia)
        {
        f << "<moment_inertia num=\"" << N << "\">" << "\n";

        for (unsigned int group_idx = 0; group_idx < N; ++group_idx)
            {
            const unsigned int tag = m_group->getMemberTag(group_idx);
            auto it = map.find(tag);
            Scalar3 I = vec_to_scalar3(snapshot.inertia[it->second]);

            f << I.x << " " << I.y << " " << I.z << "\n";
            if (!f.good())
                {
                m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
                throw runtime_error("Error writing HOOMD dump file");
                }
            }

        f << "</moment_inertia>" << "\n";
        }

    // only write the topology when the group size is the same as the system size
    if (N == m_pdata->getNGlobal())
        {
        // if the bond flag is true, output the bonds to the xml file
        if (m_output_bond)
            {
            f << "<bond num=\"" << bdata_snapshot.groups.size() << "\">" << "\n";
            std::shared_ptr<BondData> bond_data = m_sysdef->getBondData();

            // loop over all bonds and write them out
            for (unsigned int i = 0; i < bdata_snapshot.groups.size(); i++)
                {
                BondData::members_t bond = bdata_snapshot.groups[i];
                unsigned int bond_type = bdata_snapshot.type_id[i];
                f << bond_data->getNameByType(bond_type) << " " << bond.tag[0] << " " << bond.tag[1] << "\n";
                }

            f << "</bond>" << "\n";
            }

        // if the angle flag is true, output the angles to the xml file
        if (m_output_angle)
            {
            f << "<angle num=\"" << adata_snapshot.groups.size() << "\">" << "\n";
            std::shared_ptr<AngleData> angle_data = m_sysdef->getAngleData();

            // loop over all angles and write them out
            for (unsigned int i = 0; i < adata_snapshot.groups.size(); i++)
                {
                AngleData::members_t angle = adata_snapshot.groups[i];
                unsigned int angle_type = adata_snapshot.type_id[i];
                f << angle_data->getNameByType(angle_type) << " " << angle.tag[0]  << " " << angle.tag[1] << " " << angle.tag[2] << "\n";
                }

            f << "</angle>" << "\n";
            }

        // if dihedral is true, write out dihedrals to the xml file
        if (m_output_dihedral)
            {
            f << "<dihedral num=\"" << ddata_snapshot.groups.size() << "\">" << "\n";
            std::shared_ptr<DihedralData> dihedral_data = m_sysdef->getDihedralData();

            // loop over all angles and write them out
            for (unsigned int i = 0; i < ddata_snapshot.groups.size(); i++)
                {
                DihedralData::members_t dihedral = ddata_snapshot.groups[i];
                unsigned int dihedral_type = ddata_snapshot.type_id[i];
                f << dihedral_data->getNameByType(dihedral_type) << " " << dihedral.tag[0]  << " " << dihedral.tag[1] << " "
                << dihedral.tag[2] << " " << dihedral.tag[3] << "\n";
                }

            f << "</dihedral>" << "\n";
            }

        // if improper is true, write out impropers to the xml file
        if (m_output_improper)
            {
            f << "<improper num=\"" << idata_snapshot.groups.size() << "\">" << "\n";
            std::shared_ptr<ImproperData> improper_data = m_sysdef->getImproperData();

            // loop over all angles and write them out
            for (unsigned int i = 0; i < idata_snapshot.groups.size(); i++)
                {
                ImproperData::members_t improper = idata_snapshot.groups[i];
                unsigned int improper_type = idata_snapshot.type_id[i];
                f << improper_data->getNameByType(improper_type) << " " << improper.tag[0]  << " " << improper.tag[1] << " "
                << improper.tag[2] << " " << improper.tag[3] << "\n";
                }

            f << "</improper>" << "\n";
            }

        // if constraint is true, write out constraints to the xml file
        if (m_output_constraint)
            {
            f << "<constraint num=\"" << cdata_snapshot.groups.size() << "\">" << "\n";

            // loop over all angles and write them out
            for (unsigned int i = 0; i < cdata_snapshot.groups.size(); i++)
                {
                ConstraintData::members_t constraint = cdata_snapshot.groups[i];
                Scalar constraint_dist = cdata_snapshot.val[i];
                f << constraint.tag[0]  << " " << constraint.tag[1] << " " << constraint_dist << "\n";
                }

            f << "</constraint>" << "\n";
            }
        }

    f << "</configuration>" << "\n";
    f << "</hoomd_xml>" << "\n";

    if (!f.good())
        {
        m_exec_conf->msg->error() << "dump.xml: I/O error while writing HOOMD dump file" << endl;
        throw runtime_error("Error writing HOOMD dump file");
        }

    f.close();

    }

/*! \param timestep Current time step of the simulation
    Writes a snapshot of the current state of the ParticleData to a hoomd_xml file.
*/
void HOOMDDumpWriter::analyze(unsigned int timestep)
    {
    if (m_prof)
        m_prof->push("Dump XML");

    if (m_mode_restart)
        {
        string tmp_file = m_base_fname + string(".tmp");
        writeFile(tmp_file, timestep);
#ifdef ENABLE_MPI
        // only the root processor writes the output file
        if (m_pdata->getDomainDecomposition() && ! m_exec_conf->isRoot())
            {
            if (m_prof)
                m_prof->pop();
        return;
            }
#endif
        if (rename(tmp_file.c_str(), m_base_fname.c_str()) != 0)
            {
            m_exec_conf->msg->error() << "dump.xml: Error renaming restart file." << endl;
            throw runtime_error("Error writing restart file");
            }
        }
    else
        {
        ostringstream full_fname;
        string filetype = ".xml";

        // Generate a filename with the timestep padded to ten zeros
        full_fname << m_base_fname << "." << setfill('0') << setw(10) << timestep << filetype;
        writeFile(full_fname.str(), timestep);
        }

    if (m_prof)
        m_prof->pop();
    }

void export_HOOMDDumpWriter(py::module& m)
    {
    py::class_<HOOMDDumpWriter, std::shared_ptr<HOOMDDumpWriter> >(m,"HOOMDDumpWriter",py::base<Analyzer>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::string, std::shared_ptr<ParticleGroup>, bool >())
    .def("setOutputPosition", &HOOMDDumpWriter::setOutputPosition)
    .def("setOutputImage", &HOOMDDumpWriter::setOutputImage)
    .def("setOutputVelocity", &HOOMDDumpWriter::setOutputVelocity)
    .def("setOutputMass", &HOOMDDumpWriter::setOutputMass)
    .def("setOutputDiameter", &HOOMDDumpWriter::setOutputDiameter)
    .def("setOutputType", &HOOMDDumpWriter::setOutputType)
    .def("setOutputBody", &HOOMDDumpWriter::setOutputBody)
    .def("setOutputBond", &HOOMDDumpWriter::setOutputBond)
    .def("setOutputAngle", &HOOMDDumpWriter::setOutputAngle)
    .def("setOutputDihedral", &HOOMDDumpWriter::setOutputDihedral)
    .def("setOutputImproper", &HOOMDDumpWriter::setOutputImproper)
    .def("setOutputConstraint", &HOOMDDumpWriter::setOutputConstraint)
    .def("setOutputAccel", &HOOMDDumpWriter::setOutputAccel)
    .def("setOutputCharge", &HOOMDDumpWriter::setOutputCharge)
    .def("setOutputOrientation", &HOOMDDumpWriter::setOutputOrientation)
    .def("setOutputAngularMomentum", &HOOMDDumpWriter::setOutputAngularMomentum)
    .def("setOutputMomentInertia", &HOOMDDumpWriter::setOutputMomentInertia)
    .def("setVizSigma", &HOOMDDumpWriter::setVizSigma)
    .def("writeFile", &HOOMDDumpWriter::writeFile)
    ;
    }
